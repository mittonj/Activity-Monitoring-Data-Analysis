import re
import sys
from os.path import basename, splitext

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F, types
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.window import Window
from statsmodels.nonparametric.smoothers_lowess import lowess
# from fastdtw import fastdtw


# spark = SparkSession.builder.appName('final project').getOrCreate() # type: ignore
# HACK: https://stackoverflow.com/a/53620273
#       not the recommended way of allocating memory, but easier to debug
# Create a SparkSession and set logging level
spark = (SparkSession.builder.appName('final project') # type: ignore
         .config("spark.driver.memory", "3g").getOrCreate())
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

# Define the schema for the sensor data
schema = types.StructType([
    types.StructField('time', types.DoubleType()),
    types.StructField('gFx', types.DoubleType()), # g-Force in the x direction
    types.StructField('gFy', types.DoubleType()), # g-Force in the y direction
    types.StructField('gFz', types.DoubleType()), # g-Force in the z direction
    types.StructField('ax', types.DoubleType()), # Linear Acceleration in the x direction
    types.StructField('ay', types.DoubleType()), # Linear Acceleration in the y direction
    types.StructField('az', types.DoubleType()), # Linear Acceleration in the z direction
    types.StructField('wx', types.DoubleType()), # Gyroscope in the x direction
    types.StructField('wy', types.DoubleType()), # Gyroscope in the y direction
    types.StructField('wz', types.DoubleType()), # Gyroscope in the z direction
    types.StructField('Latitude', types.DoubleType()), # Latitude
    types.StructField('Longitude', types.DoubleType()), # Longitude
    types.StructField('Speed', types.DoubleType()), # Speed (m/s)
])

# Function to extract the activity type (walk, run, skate) from the filename using regex
def get_classifier(path):
    match = re.search(r'(run|skate|walk)', path)
    return match.group(1) if match else None

# Define a User-Defined Function (UDF) to apply the get_classifier function to the 'filename' column
classifier = F.udf(get_classifier, returnType=types.StringType())

def add_statistical_features(data):
    data = data.withColumn('gF_mag', F.sqrt(F.col('gFx')**2 + F.col('gFy')**2 + F.col('gFz')**2))
    data = data.withColumn('a_mag', F.sqrt(F.col('ax')**2 + F.col('ay')**2 + F.col('az')**2))
    data = data.withColumn('w_mag', F.sqrt(F.col('wx')**2 + F.col('wy')**2 + F.col('wz')**2))
    return data

def add_walking_features(data):
    data = data.withColumn('distance', F.col('Speed') * F.col('sampling_interval'))
    return data

# Function to generate plots for each feature against time and save them in the specified output directory
def graph_data(df, out_directory):
    # Graph all data
    filenames = df.select('filename', 'activity').distinct().collect()

    sns.set(style='whitegrid')

    # Loop through each filename

    walk_i = 0
    run_i = 0
    skate_i = 0

    for row in filenames:
        filename = row['filename']
        activity = row['activity']

        if(activity == 'walk'):
            walk_i += 1
        elif(activity == 'run'):
            run_i += 1
        elif(activity == 'skate'):
            skate_i += 1

        # Filter the DataFrame for the specific filename
        df_filename = df.filter(df['filename'] == filename).toPandas()

        feat_cols = [
            'activity',
            'filename',
            'minute',
            'max_minute',
            'sampling_interval',
            'change_filename',
            'filename_increment',
            'max_time',
            'row_index',
            'time',
        ]

        # Create a plot for each feature against time
        for feature in df.columns[1:]:  # Skip the 'filename' column
            if feature in feat_cols:
                continue

            plt.figure(figsize=(30, 10))
            sns.lineplot(data=df_filename, x='time', y=feature)            
            plt.xlabel('Time')
            plt.ylabel(feature)
            plt.title(f'{feature} vs. Time')
            if(activity == 'walk'):
                print(f'{out_directory}-graphs/{activity}/{walk_i}_{feature}.svg')
                plt.savefig(f'graphs/walk/walk{walk_i}_{feature}.svg', format='svg')
            elif(activity == 'run'):
                print(f'{out_directory}-graphs/{activity}/{run_i}_{feature}.svg')
                plt.savefig(f'graphs/run/run{run_i}_{feature}.svg', format='svg')
            elif(activity == 'skate'):
                print(f'{out_directory}-graphs/{activity}/{skate_i}_{feature}.svg')
                plt.savefig(f'graphs/skate/skate{skate_i}_{feature}.svg', format='svg')
            plt.close()

# Function to group data by 'filename' and 'interval' and calculate the average of various sensor attributes within each group
def group_interval(df):

    # Grouping for John & Vinh's models

    df = df.groupBy('filename', 'interval').agg(
        F.avg('gFx').alias('avg_gFx'),
        F.avg('gFy').alias('avg_gFy'),
        F.avg('gFz').alias('avg_gFz'),
        F.avg('ax').alias('avg_ax'),
        F.avg('ay').alias('avg_ay'),
        F.avg('az').alias('avg_az'),
        F.avg('wx').alias('avg_wx'),
        F.avg('wy').alias('avg_wy'),
        F.avg('wz').alias('avg_wz'),
        F.avg('Latitude').alias('avg_Latitude'),
        F.avg('Longitude').alias('avg_Longitude'),
        F.avg('Speed').alias('avg_Speed'),
        F.avg('gF_mag').alias('avg_gF_mag'),
        F.avg('a_mag').alias('avg_a_mag'),
        F.avg('w_mag').alias('avg_w_mag'),
        F.avg('distance').alias('avg_distance'),
        F.first('activity').alias('activity')
    )

    df = df.drop('interval', 'avg_Latitude', 'avg_Longitude')
    return df

# Main function to preprocess and export the data (to json or csv)
def main(in_directory, out_directory, min_activity_duration: int = 60,
         to_jsonl: bool = False):

    #
    ## load data
    #
    # only the basename w/o extension: apple/ball.c -> ball
    f_data_name = F.udf(
        lambda x: splitext(basename(x))[0],
        returnType = types.StringType())
    df = spark.read.csv(in_directory, schema=schema).withColumn('filename', f_data_name(F.input_file_name()))

    # Add a new column 'row_index' that resets for each 'filename' group
    windowSpec = Window.partitionBy('filename').orderBy(F.monotonically_increasing_id())
    df = df.withColumn('row_index', F.row_number().over(windowSpec))

    df = df.withColumn('activity', classifier(df['filename']))

    # Remove all entries where speed is 0
    # This removes all entries where the user is not moving, whether it be for setup or for a break
    df = df.filter(df['speed'] > 0)

    sampling_interval_threshold = 1   # sec

    # Add a new column time difference that calculates the difference between the current row and the previous row
    df = df.withColumn('sampling_interval', F.col('time') - F.lag('time', 1).over(Window.partitionBy('filename').orderBy('time')))
    df = df.withColumn('change_filename', F.when(F.col('sampling_interval') > sampling_interval_threshold, 1).otherwise(0))
    df = df.withColumn('filename_increment', F.sum('change_filename').over(Window.partitionBy('filename').orderBy('time')))
    df = df.withColumn('filename', F.concat(df['filename'], F.lit('_'), F.col('filename_increment')))

    # NOTE: `time` attrib is *NOT* unique, but `(filename, time)` pair is
    df = df.withColumn('time', F.col('time') - F.min('time').over(Window.partitionBy('filename')))

    # Filter and keep only those filenames where the maximum time is greater than 60 seconds
    max_time_df = df.groupBy('filename').agg(F.max('time').alias('max_time'))
    filenames_to_keep = max_time_df.filter(max_time_df['max_time'] > min_activity_duration).select('filename')
    df = df.join(filenames_to_keep, on='filename', how='inner')
    df = df.withColumn('interval', F.col('time') / min_activity_duration)
    df = df.withColumn('max_minute', F.max('interval').over(Window.partitionBy('filename')))
    df = df.filter(df['interval'] < F.floor(df['max_minute']))
    df.drop('max_minute')
    df = df.withColumn('interval', F.floor(df['interval']))

    df = df.drop('max_time', 'change_filename', 'filename_increment', 'max_minute', 'row_index')
    # df = df.cache()
    #
    ## remove outliers
    #

    # HACK: removes invalid intervals that were produced by time-shift.
    #       It should be done by removing the first row from each loaded file
    #       after we computed the sampling interval. But this is kinda tricky to
    #       do with only spark: https://stackoverflow.com/a/57532838
    df = df.filter((df['sampling_interval'] < 0.9) & ~df['sampling_interval'].isNull())

    # # remove outliers: g-force only within [6, 8]
    # # TODO: maybe we should not filter gforce this way
    df = df.withColumn("gF_magnitude", F.sqrt(F.col("gFx")**2 + F.col("gFy")**2 + F.col("gFz")**2))
    # df = df.filter((F.col("gF_magnitude") >= 6) & (F.col("gF_magnitude") <= 8))

    
    ## data smoothing
    
    # NOTE: this might take a while to smooth data; comment it out if is not
    #       needed or for testing.

    df = df.cache()

    average_value = df.agg(F.avg("sampling_interval")).collect()[0][0]
    @pandas_udf(types.DoubleType()) # type: ignore
    def smooth_data(col: pd.Series) -> pd.Series:
        return pd.Series(lowess(col, np.arange(len(col)), frac=average_value, return_sorted=False))
    df = df.withColumn('gFx_smoothed', smooth_data(df['gFx']))
    df = df.withColumn('gFy_smoothed', smooth_data(df['gFy']))
    df = df.withColumn('gFz_smoothed', smooth_data(df['gFz']))

    df = df.withColumn('ax_smoothed', smooth_data(df['ax']))
    df = df.withColumn('ay_smoothed', smooth_data(df['ay']))
    df = df.withColumn('az_smoothed', smooth_data(df['az']))

    df = df.withColumn('wx_smoothed', smooth_data(df['wx']))
    df = df.withColumn('wy_smoothed', smooth_data(df['wy']))
    df = df.withColumn('wz_smoothed', smooth_data(df['wz']))



    #
    ## scaling attributes
    #
    cols_to_scale = ['gFx', 'gFy', 'gFz', 'ax', 'ay', 'az', 'wx', 'wy', 'wz']

    for col in cols_to_scale:
        mean_value = df.agg(F.mean(col)).collect()[0][0]
        df = df.withColumn(col, F.col(col) - mean_value)

    for col in cols_to_scale:
        std_value = df.agg(F.stddev(col)).collect()[0][0]
        df = df.withColumn(col, F.col(col) / std_value)

    df = df.cache()
    # graph_data(df, out_directory)

    
    # Feature Engineering
    # NOTE: this is a very simple feature engineering, and it can be improved
    df = add_statistical_features(df)

    df = add_walking_features(df)

    #
    ## export
    #
    if to_jsonl:
        df.write.json(out_directory, mode='overwrite')
    else:
        #
        ## final grouping (for some ML algorithms)
        #
        df = group_interval(df)
        df.write.csv(out_directory, mode='overwrite', header=True)

# Check if the script is being run as the main module and execute the main function
if __name__ == '__main__':
    in_directory = "data/all_data"
    min_activity_duration = 60

    use_jsonl = False
    if len(sys.argv) >= 3:
        in_directory = sys.argv[1]
        min_activity_duration = int(sys.argv[2])
        if len(sys.argv) >= 4:
            use_jsonl = bool(sys.argv[3])
    else:
        print(f"Using the default in_directory: {in_directory}")

    out_directory = f'{in_directory}-oput_{min_activity_duration}s'
    main(in_directory, out_directory, min_activity_duration, use_jsonl)
