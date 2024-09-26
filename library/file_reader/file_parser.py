import pandas as pd
import numpy as np
from math import pi
from datetime import datetime, timedelta
from struct import unpack
import json
from asammdf import MDF
from io import BytesIO
from library.file_reader.dictionary_funcs import get_dictionary


def create_df(filename, sampling_rate='1Hz', upload_type="upload", datetime_label="timestamp", timestamp_format='Unix', sampling_type='mean', csv_seperator=';', csv_skiplines=0, dictionary='<None>'):

    # GET FILE EXTENSION
    file_type = filename.name.split(".")[-1]
    if file_type == "gzip": # for parquet gzip
        file_type = filename.name.split(".")[-2]

    print("LOG  :: file_parser ::  The filetype is", file_type)

    # READ FILE BASED ON FILE TYPE
    try:  # if reading fails return None
        if file_type == 'parquet' or file_type == 'gzip':  # also includes parquet.gzp
            df = pd.read_parquet(filename)
        elif file_type == 'csv':
            if csv_seperator == 'tab':
                csv_seperator = '\t'
            df = pd.read_csv(filename, sep=csv_seperator, skiprows=csv_skiplines, encoding='latin-1')

            #try:
            # df = pd.read_csv(filename, sep=csv_seperator, skiprows=csv_skiplines, encoding='unicode_escape', engine="pyarrow")
            #except UnicodeDecodeError:
            # READ CHUNKS
            # df = pd.DataFrame()
            # for chunk in pd.read_csv(filename, sep=csv_seperator, skiprows=csv_skiplines, chunksize=1000):
            #     df = pd.concat([df, chunk], ignore_index=True)
        elif file_type == 'dat':
            df = pd.read_csv(filename)
        elif file_type == 'xlsx':
            df = pd.read_excel(filename)
        elif file_type == 'mf4':
            if upload_type == "upload":
                bytes_data = BytesIO(filename.getvalue())
                mdf = MDF(bytes_data)
                df = mdf.to_dataframe()
            else:
                mdf = MDF(filename)
                df = mdf.to_dataframe()
        else:
            return None
    except Exception as e:
        print('LOG :: file_parser :: Error reading file with pandas ::', e)
        return None


    # DICTIONARY RENAME
    if dictionary != '<None>':
        feature_dict = get_dictionary(dictionary)
        unique_features = list(set(feature_dict.values()))
        df = df.rename(columns=feature_dict)
        columns_to_keep = list(set(df.columns.to_list()).intersection(unique_features))
        columns_to_keep.sort()
        df = df[columns_to_keep]

    # DATETIME
    if datetime_label in list(df.columns):
        # 1 Hz / Second Resampling
        # df = df.groupby(datetime_label, as_index=False).mean()   # 1 Hz Resampling
        print(df.head(1))
        # First Resampling to avoid duplicates
        # df = df.groupby('DateTime', as_index=False).mean()   # 1 Hz Resampling
        if timestamp_format == 'Unix':
            # df['DateTime'] = df[datetime_label].apply(lambda x: datetime.utcfromtimestamp(x))
            try:
                df['DateTime'] = pd.to_datetime(df[datetime_label], unit="us").astype('datetime64[s]')
            except:
                return None
        else:
            df['DateTime'] = df[datetime_label]


        # FIRST RESAMPLING TO AVOID DUPLICATES
        if sampling_type == 'mean':
            df = df.resample('s', on='DateTime').mean()
        elif sampling_type == 'first':
            df = df.resample('s').first()
        else:  # max
            df = df.resample('s').max()


    if 'DateTime' in list(df.columns):
        df = df.groupby('DateTime', as_index=False).first()  # 1 Hz Resampling

    # DROP DUPLICATE COLUMN NAMES
    df = df.iloc[:, ~df.columns.duplicated()]

    # # FILL NULL BACKWARDS
    # df.fillna(method="bfill", inplace=True)
    #
    # # FILL NULL FORWARD
    # df.fillna(method="ffill", inplace=True)

    # REDUCE DF SIZE
    # FLOAT 64 -> 32 Bits
    float64_cols = list(df.select_dtypes(include='float64'))  # Select columns with 'float64' dtype
    df[float64_cols] = df[float64_cols].astype('float32')  # The same code again calling the columns

    # TREAT BAD VALUES
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # df[float64_cols] = df[float64_cols].round(2)  # round to 2 decimals
    # INT 64 -> 16 Bits
    # int64_cols = list(df.select_dtypes(include='int64'))  # Select columns with 'int64' dtype
    # df[int64_cols] = df[int64_cols].astype('int32')  # The same code again calling the columns

    # print(df.info(memory_usage='deep')) # TEST
    # print('ALL END', time.time() - start) # SPEED TEST
    return df

