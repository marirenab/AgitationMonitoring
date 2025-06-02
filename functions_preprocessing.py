import pandas as pd
import numpy as np


# I extend the labels to 6 days before (7 days including the recording)
days_before = 7

def expand_days_before(df):
    date = df['date'].values[0]
    df = [df] * (days_before + 1)
    new_date_values = np.arange(-days_before, 1)
    new_dates = [
        date + pd.Timedelta(days=d) for d in new_date_values
    ]
    df = pd.concat(df)
    df['date'] = new_dates
    return df


groupby_columns = ['patient_id', 'date']

map_location = {
            'bathroom1': 'Bathroom', 
            'WC1': 'Bathroom',
            'kitchen': 'Kitchen',
            'hallway': 'Lounge',
            'corridor1': 'Lounge',
            'dining room': 'Kitchen',
            'living room': 'Lounge',
            'lounge': 'Lounge',
            'study': 'Lounge',
            'office': 'Lounge',
            'conservatory': 'Lounge',
            'bedroom1': 'Bedroom'
        }

def align_sensor_activity_data(sensor_data, activity, labels=None):
    # Deep copy of light and activity dataframes
    sensor_data = sensor_data.copy(deep=True)
    activity = activity.copy(deep=True)
    
    if labels is not None:
        filter = labels[["date", "patient_id", "event"]]
        # Merge labels with light and activity data
        sensor_data = pd.merge(filter, sensor_data, on=["patient_id", "date"])
        activity = pd.merge(activity, filter, on=["patient_id", "date"])
        sensor_data = sensor_data[["start_date","patient_id","location_name","hour","date","value","event"]]
        
    # Merge light and activity data
        new = pd.merge(sensor_data, activity, on=["patient_id", "date", "hour", "location_name","event"])
    else:
        new= pd.merge(sensor_data, activity, on=["patient_id", "date", "hour", "location_name"])
    # Process combined data
    new = new.drop(columns=["start_date_y"]) #Dropping the activity start_date
    new = new[["start_date_x", "date", "patient_id", "location_name", "value"]]

    new = new.drop_duplicates(subset=["start_date_x", "patient_id", "value"]) #There are many duplicate sensor observations within an hour from single room so removing those
    new['start_date_x'] = pd.to_datetime(new['start_date_x'])
    new['hour'] = new['start_date_x'].dt.hour
    new['time_period'] = pd.cut(new['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'], include_lowest=True, right=False)
    new = new.drop(columns=["start_date_x"])
    new = new.groupby(["patient_id","date","time_period","location_name"]).mean().reset_index()
    #Mapping the locations into bigger ones that are common accross all participants
    new["location"]= new["location_name"].replace(map_location)

    return new 

def process_data_imputation(sensor_data, labels=None):
    # Deep copy of the sensor_data dataframe
    sensor_data = sensor_data.copy(deep=True)
    
    # Create time_period column based on hour
    sensor_data['time_period'] = pd.cut(sensor_data['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'], include_lowest=True, right=False)
    
    # Map location names
    sensor_data["location_2"] = sensor_data["location_name"].map(map_location)
    
    # Exclude rows with matching (date, patient_id) in labels
    if labels is not None:
        sensor_data = sensor_data[~sensor_data.set_index(["date", "patient_id"]).index.isin(labels.set_index(["date", "patient_id"]).index)]
    else:
        sensor_data = sensor_data
    # Select relevant columns
    sensor_data = sensor_data[["patient_id", "date", "value", "location_name", "location_2", "time_period"]]
    
    # Drop rows where location_2 is NaN
    sensor_data = sensor_data.dropna(subset=["location_2"]) # we do not want NaNs since we are training the imputer with this df
    
    # Group by patient_id, location_name, time_period and resample by 8 days, then calculate mean
    sensor_data = sensor_data.groupby(['patient_id', 'location_name', 'time_period']).resample('8D', on='date').mean()
    
    # Reset index and drop rows with NaN values
    sensor_data = sensor_data.reset_index().dropna()
    
    return sensor_data


def personalised_imputation_PIR(new_df, sensor_data, knn_imputer, missing_patients,df_pivot_new):
    new_df["date"] = new_df["event"]
    imputed_dfs = []  # Create an empty list to store the imputed DataFrames
    for patient in missing_patients:
        data = new_df[new_df["patient_id"] == patient]
        sensor_data_patient = sensor_data[sensor_data["patient_id"] == patient]
        sensor_data_patient = sensor_data_patient[["date", "location_name", "time_period", "value"]]
        sensor_data_patient = sensor_data_patient.pivot_table(index=["date"], columns=["location_name", "time_period"], values="value", aggfunc="mean")
        sensor_data_patient.columns = ['_'.join(col) for col in sensor_data_patient.columns]
        sensor_data_patient = sensor_data_patient.dropna(axis=1, how="all")
        patient_data = pd.concat([sensor_data_patient, data], join='outer')
        data = data.set_index(["patient_id","date"])
        patient_data = patient_data.dropna(axis=1, how='all')
        patient_data = patient_data.set_index(['patient_id','date'])
        transformed_columns = sensor_data_patient.columns
        knn_imputer.fit(sensor_data_patient[transformed_columns])
        # Transform and create a DataFrame with the transformed columns only
        patient_data_transformed = pd.DataFrame(knn_imputer.transform(patient_data[transformed_columns]), 
                                                index=patient_data.index, 
                                                columns=transformed_columns)

        patient_data = patient_data.drop(columns=transformed_columns)  # Drop the original columns
        patient_data = pd.concat([patient_data, patient_data_transformed], axis=1)
        patient_data = patient_data.reset_index()
        patient_data = patient_data[patient_data["patient_id"] == patient]
        imputed_dfs.append(patient_data)
    
    # Concatenate imputed dataframes and return
   
    imputed_df = pd.concat(imputed_dfs)
    imputed_df = imputed_df.dropna(subset=["patient_id"])
    columns_to_drop = ["index", "date"]
    columns_to_drop_existing = [col for col in columns_to_drop if col in imputed_df.columns]
    if columns_to_drop_existing:
        imputed_df.drop(columns=columns_to_drop_existing, inplace=True)
    df_impute_mean = pd.concat([df_pivot_new, imputed_df])
        
        # Calculate the number of NaN values in each row
    nan_counts = df_impute_mean.isnull().sum(axis=1)
        
        # Add the 'nan_counts' column to the DataFrame
    df_impute_mean = df_impute_mean.assign(nan_counts=nan_counts)
    df_impute_mean = df_impute_mean.sort_values(by=["nan_counts"])
    df_impute_mean = df_impute_mean.drop_duplicates(["patient_id", "event"], keep="first")
    df_impute_mean = df_impute_mean.drop(columns=["nan_counts"])
    
    if "index" in df_impute_mean.columns:
        df_impute_mean.drop(columns=["index"], inplace=True)


    return df_impute_mean


def general_imputation(df_impute_mean, sensor_data, iterative_imputer,new):
    df_impute_mean = df_impute_mean.set_index(["patient_id","event"])
    df = df_impute_mean.stack()
    df = df.reset_index()
    df[['location_name', 'time_period']] = df['level_2'].str.split('_', n=1, expand=True)
    df = df.drop('level_2', axis=1)
    df.rename(columns={0:"value"},inplace=True)
    keep_event = new[["mood","patient_id","event"]]
    df_try = pd.merge(df, keep_event, on=["patient_id","event"])
    df_try["location"] = df_try["location_name"].map(map_location)
    df_try = df_try.drop_duplicates(subset=["patient_id","value","time_period","location_name","location","event"],keep="first")
    df_try = df_try.groupby(["patient_id", "mood", "event", "time_period", "location"]).mean()
    df_try = df_try.unstack(level=[ "location","time_period"])
    df_try_rename = df_try.copy(deep=True)
    df_try_rename.columns = [f'{loc}_{time}_mean' for value, loc, time in df_try_rename.columns]

    rows_with_nans = df_try_rename[df_try_rename.isna().any(axis=1)]

    sensor_data["location"] = sensor_data["location_name"].map(map_location)
    sensor_data_impute = sensor_data.pivot_table(index=["patient_id", "date"], columns=["location", "time_period"], values="value", aggfunc="mean")
    sensor_data_impute = sensor_data_impute.drop(sensor_data_impute[sensor_data_impute.isnull().sum(axis=1) > 2].index)
    sensor_data_impute = sensor_data_impute.stack().reset_index()
    sensor_data_impute = sensor_data_impute.drop(sensor_data_impute[sensor_data_impute.isnull().sum(axis=1) > 2].index)
    sensor_data_impute = sensor_data_impute.set_index(["patient_id", "date", "time_period"])

    # Perform iterative imputation on light data

    iterative_imputer.fit(sensor_data_impute)
    patient_impute = df_try.stack()
    columns_to_impute = [('value', 'Bathroom'), ('value', 'Bedroom'), ('value', 'Kitchen'), ('value', 'Lounge')]
    names = ['Bathroom', 'Bedroom', 'Kitchen', 'Lounge']
    patient_impute_imp = pd.DataFrame(iterative_imputer.transform(patient_impute[columns_to_impute]), columns=names, index=patient_impute.index)
    return patient_impute_imp


def calculate_ratio(weather, labels_keep, sensor_imputed, column_name, ratio_column_name,sensor_indoor_mean):
    # Make a deep copy of the weather data
    weather = weather.copy(deep=True)
    
    # Merge labels_keep with weather on the 'date' column
    df = pd.merge(labels_keep, weather, on=["date"])
    
    # Select specific columns
    df = df[["patient_id", "mood", "date", "event", column_name]]
    
    # Group by patient_id, mood, and event and calculate the mean for each group
    df = df.groupby(["patient_id", "mood", "event"]).mean()
    
    # Reset index to make 'patient_id', 'mood', and 'event' columns again
    df = df.reset_index()
    
    # Create a new DataFrame with specific columns
    df_ratio = df[["patient_id", "mood", "event", column_name]]
    
    # Merge sensor_imputed with df_ratio on 'patient_id', 'mood', and 'event'
    sensor_imputed = pd.merge(sensor_imputed, df_ratio, on=["patient_id", "mood", "event"])
    
    # Calculate the ratio of indoor mean illuminance to the column specified
    sensor_imputed[ratio_column_name] = sensor_imputed[sensor_indoor_mean] / sensor_imputed[column_name]
    
    # Drop the specified column and 'temp' from the DataFrame
    sensor_imputed.drop(columns=[column_name], inplace=True)
    
    return sensor_imputed


def calculate_absence_activity(act, doors, bed, weather_first, labels_keep=None, patients_labeled):
    # Drop unnecessary columns and add a "count" column to 'act'
    act = act.drop(columns=["source", "location_id", "home_id"])
    act["count"] = 1
    
    # Group by patient_id and start_date, summing the counts
    act_sum = act.groupby(["patient_id", "start_date"]).sum().reset_index()

    # Resample data per minute and concatenate
    concat = pd.DataFrame()
    for patient in patients_labeled:
        data_patient = act_sum[act_sum["patient_id"] == patient]
        resample_patient = data_patient.set_index("start_date").resample('1T').asfreq()
        resample_patient["patient_id"] = patient
        concat = pd.concat([concat, resample_patient])
        
    # Fill NaN values with 1000
    concat = concat.reset_index()
    concat["date"] = concat["start_date"].dt.date
    concat.groupby(["patient_id","date"])["count"].unique()
    concat = concat.fillna(1000)
    
    # Keep groups that have only 1000 in 'count' column
    filtered_groups = concat.groupby(['patient_id', 'date']).filter(lambda group: not (group['count'] == 1000).all())

    filtered = filtered_groups[filtered_groups["count"] == 1000]
    filtered["count"] = filtered["count"].replace(1000,1)
    filtered["hour"] =filtered["start_date"].dt.time
    filtered = filtered.drop(columns=["start_date"])
    filtered["date"] = pd.to_datetime(filtered["date"])

# Filtered contains t he dates where there was no activity in the house for some time. 
    
    if labels_keep is not None:
        # Merge with labels_keep
        filtered = pd.merge(labels_keep, filtered, on=["patient_id","date"])
    else:
        filtered= filtered

    # Filter 'doors'
    doors = doors[(doors["location_name"] == "main door") | (doors["location_name"] == "back door")]
    doors = doors[doors["patient_id"].isin(act["patient_id"].unique())]
    doors["hours"] = doors["dur"] / 3600
    doors["date"] = doors["start_date"].dt.date
    doors["hour"] = doors["start_date"].dt.hour
    doors['time_period'] = pd.cut(doors['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'], include_lowest=True, right=False)
    doors["date"] = pd.to_datetime(doors["date"])

    # Merge with 'labels_keep'
    if labels_keep is not None:
        df = pd.merge(doors, labels_keep, on=["patient_id", "date"])
        df = df[["patient_id", "dur", "hour", "time_period", "date", "mood"]]  

    else:
        df = doors
        df = df[["patient_id", "dur", "hour", "time_period", "date"]]

     
    # Filtering only the dates where inactivity was between the sunrise and sunset
    if labels_keep is not None:
        df_new = pd.merge(weather_first, labels_keep, on=["date"])
        df_new = df_new[["sunrise_hour", "sunset_hour", "date", "mood", "event", "patient_id", "illuminance", "cloudcover", "tempmin", "tempmax", "temp", "feelslikemin", "feelslikemax", "feelslike", "uvindex", "visibility"]]
        df_absence = pd.merge(df_new, filtered, on=["date", "patient_id", "mood"])

    else:
        df_new = weather_first
        df_new = df_new[["sunrise_hour", "sunset_hour", "date", "illuminance", "cloudcover", "tempmin", "tempmax", "temp", "feelslikemin", "feelslikemax", "feelslike", "uvindex", "visibility"]]
        df_absence = pd.merge(df_new, filtered, on=["date"])
    
    
 
    df_absence = df_absence[df_absence['hour'].between(df_absence['sunrise_hour'], df_absence['sunset_hour'])]

    
        # Filtering only when the times of inactivity that were between sunrise and sunset did not overlap with times when patient was in bed
    bed = bed[["patient_id", "datetime", "start_time", "end_time"]]
    bed["end_time"] = bed["end_time"].astype(str)
    bed["end_time"] = bed["end_time"].apply(london_to_utc)  # Converting to UTC since profile domain has hours in local time
    bed["end_time"] = pd.to_datetime(bed["end_time"])
    bed["date"] = bed["end_time"].dt.date
    bed["end_time"] = bed["end_time"].dt.time
    bed = bed.drop(columns=["start_time"])
    bed["date"] = pd.to_datetime(bed["date"])
    df_absence = pd.merge(df_absence, bed, on=["date", "patient_id"])
    df_absence = df_absence[df_absence['hour'] > df_absence['end_time']]

    if labels_keep is not None:
        duration_absence = df_absence.groupby(["patient_id", "date", "mood"])["count"].sum()

    else:
        duration_absence = df_absence.groupby(["patient_id", "date"])["count"].sum()
    
    duration_absence = duration_absence.reset_index()
 
    duration_absence["hours"] = duration_absence["count"] / 60
    df_absence_verify = df_absence[["date", "patient_id", "hour"]]

        # Convert date columns to datetime objects
    doors["start_date_day"] = pd.to_datetime(doors["start_date"].dt.date)
    doors["end_date_day"] = pd.to_datetime(doors["end_date"].dt.date)

    # Keep only dates where start_date and end_date are the same
    doors_keep = doors[doors["start_date_day"] == doors["end_date_day"]].copy()

    # Drop unnecessary columns
    doors_keep["start_date_day"] = pd.to_datetime(doors_keep["start_date_day"])
    doors_keep["date"] = doors_keep["start_date_day"].dt.date
    doors_keep = doors_keep.drop(columns=["start_date_day","end_date_day"])
    doors_keep["start_date_min"] =doors_keep["start_date"].dt.floor('min')
    doors_keep["end_date_min"] =doors_keep["end_date"].dt.floor('min')
    df_absence_verify.rename(columns={"hour":"activity_hour"},inplace=True)
    doors_keep["date"] = pd.to_datetime(doors_keep["date"])
    df_absence_verify = df_absence_verify.sort_values(by=["patient_id","date"])
    doors_keep = doors_keep.sort_values(by=["patient_id","date"])
    doors_keep = doors_keep.set_index(["patient_id","date"])
    df_absence_verify = df_absence_verify.set_index(["patient_id","date"])
    df_verify = pd.merge(doors_keep,df_absence_verify, on=["patient_id","date"])
    doors_keep["test_hour"] = doors_keep["start_date_min"]
    df_absence_verify = df_absence_verify.reset_index()
    df_absence_verify["date"] = df_absence_verify["date"].astype(str)
    df_absence_verify["activity_hour"] = df_absence_verify["activity_hour"].astype(str)
    df_absence_verify["test_hour"] = df_absence_verify["date"] + " " + df_absence_verify["activity_hour"]
    df_absence_verify["test_hour"] = pd.to_datetime(df_absence_verify["test_hour"])

    df_absence_verify["test_hour"] = pd.to_datetime(df_absence_verify["test_hour"])
    df_absence_verify = df_absence_verify.set_index(["patient_id","date"])
    doors_keep = doors_keep.reset_index()
    df_absence_verify = df_absence_verify.reset_index()
    df_absence_verify["minutes"] = (df_absence_verify["test_hour"] - pd.to_datetime("2020-07-23 00:00:00")).dt.total_seconds() / 60
    doors_keep["minutes"] = (doors_keep["test_hour"] - pd.to_datetime("2020-07-23 00:00:00")).dt.total_seconds() // 60
    df_absence = df_absence[df_absence['hour'].between(df_absence['sunrise_hour'], df_absence['sunset_hour'])]
    df_absence_verify = df_absence_verify.reset_index()
    df_absence_verify["date"] = pd.to_datetime(df_absence_verify["date"])

    # Convert the 'test_hour' column to a numeric data type
    df_absence_verify_sorted = df_absence_verify.sort_values(by="minutes")
    doors_keep_sorted = doors_keep.sort_values(by="minutes")
    # Merge the DataFrames based on the closest hour
    merged_df = pd.merge_asof(
        df_absence_verify_sorted, doors_keep_sorted,by=["patient_id","date"],
        on=['minutes'],
        direction='nearest'
    )
    
    merged_df = merged_df.dropna()
    merged_df["end_date_min"]=merged_df["end_date_min"].dt.time
    merged_df["start_date_min"]=merged_df["start_date_min"].dt.time
    merged_df["activity_hour"] = pd.to_timedelta(merged_df["activity_hour"])
    merged_df["activity_hour"] = merged_df["activity_hour"].apply(lambda x: pd.Timestamp(0) + x).dt.time



        # We only keep the weather data when inactivity happened between the hours that the door was used
    verify = merged_df[
            (merged_df['activity_hour'] >= merged_df['start_date_min']) &
            (merged_df['activity_hour'] <= merged_df['end_date_min'])
        ]

        # Drop unnecessary columns and add a duration_absenceactivity column
    verify.drop(columns=["test_hour_x", "test_hour_y", "location_name", "source", "sink", "hours"], inplace=True)
    verify["duration_absenceactivity"] = 1  # 1 is every minute

        # Calculate the duration of time spent outside
    duration = verify.groupby(["patient_id", "date"])["duration_absenceactivity"].sum()
    duration = duration.reset_index()
    verify = verify.sort_values(by=["patient_id"])

    if labels_keep is not None:
        df_absence_new = pd.merge(duration, df_new, on=["patient_id", "date"])  #df_new has the weather data. We merge with the duration df which has the dates filtered. 
    else:
        df_absence_new = pd.merge(duration, df_new, on=["date"])  #df_new has the weather data. We merge with the duration df which has the dates filtered. 


    return df_absence_new


def calculate_ratio_stats(weather, sensor_data, activity,patients_labeled, column, prefix,ratio_column_name):
    sensor_data = sensor_data[sensor_data["patient_id"].isin(patients_labeled)]
    activity = activity[activity["patient_id"].isin(patients_labeled)]
    new = align_sensor_activity_data(sensor_data, activity)
    new_resampled = process_data_imputation(new)
    new_resampled["location"] = new_resampled["location_name"].replace(map_location)
    locations_keep = ["Bathroom","Bedroom","Kitchen","Lounge"]
    new = new[new["location"].isin(locations_keep)]
    ratio = new[["date","patient_id","location_name","time_period","value"]]
    ratio_mean = ratio[(ratio["time_period"] == "morning") | (ratio["time_period"] == "afternoon")]
    ratio_mean = ratio_mean.groupby(["date","patient_id","time_period"])["value"].mean().reset_index()
    ratio_mean = ratio_mean.dropna()
    ratio_mean = ratio_mean.groupby(["date","patient_id"])["value"].mean().reset_index()
    outdoor = weather
    outdoor["date"] = pd.to_datetime(outdoor["date"])
    outdoor = outdoor[["date",column]]
    outdoor = pd.merge(outdoor, ratio_mean, on=["date"])
    outdoor[ratio_column_name] = outdoor["value"]/outdoor[column]
    outdoor = outdoor.groupby(['patient_id']).resample('8D', on='date').mean().reset_index()
    ratio_mean = outdoor.groupby(["patient_id"])[ratio_column_name].mean().reset_index()
    ratio_std = outdoor.groupby(["patient_id"])[ratio_column_name].std().reset_index()
    sensor_indoor_mean =outdoor.groupby(["patient_id"])["value"].mean().reset_index()
    sensor_indoor_std =outdoor.groupby(["patient_id"])["value"].std().reset_index()
    sensor_indoor_mean.rename(columns={"value":f"{prefix}_indoor_mean"},inplace=True)
    sensor_indoor_std.rename(columns={"value":f"{prefix}_indoor_mean"},inplace=True)
    return sensor_indoor_mean, sensor_indoor_std , ratio_mean, ratio_std


def normalize(df, mean_scale, std_scale, columns_to_scale):
    scale_global = df[columns_to_scale]
    mean_df = mean_scale.set_index('patient_id')
    std_df = std_scale.set_index('patient_id')
    scale_global = scale_global.reset_index()
    scale_global = (scale_global.set_index('patient_id') - mean_df) / std_df
    scale_global = scale_global.dropna(subset=scale_global.columns, how='all')
    df = df.reset_index()
    df=df.set_index(["patient_id"])
    
    # Replace the scaled values in the original dataframe
    df[columns_to_scale] = scale_global[columns_to_scale].values
    df = df.reset_index()
    df = df.set_index(["patient_id","event"])
    return df
