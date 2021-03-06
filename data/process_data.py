import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sqlalchemy import create_engine, inspect
from sqlalchemy.schema import (
    MetaData,
    Table,
    DropTable
    )

def load_data(messages_filepath, categories_filepath):

    '''
    Load messages and categories dataframes
    INPUT :
        messages_filepath [string]: The path of the message's file
        categories_filepath [string]: The path of the categorie's file
    
    OUTPUT:
        messages, categories : A tuple containing a dataframe of messages (messages) and dataframe of 
                                categories

    '''

    #importing data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
        
    return messages, categories


def clean_data(df):
    
    '''
    Applies all steps of the cleaning process and generates a structured dataframe 
    INPUT :
       df [tuple]: A tuple containing a dataframe of messages (messages) and dataframe of 
                     categories
    
    OUTPUT:
        df [dataframe]: The final dataframe

    '''


    messages, categories = df
    
    # merge datasets
    df = pd.merge(messages,categories,on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
    
    # Extract a list of new column names for categories.
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x[:-2]))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')
    
    # lets change all values 2 to 1 in 'related' category
    filt = categories.related == 2
    categories.loc[filt,'related'] = 1
    
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    Save the clean dataframe to the indicated database.
    INPUT:
        df : the dataframe to save
        database_filename [string]: The database name

    OUTPUT:

    '''
    # Connect to the database
    engine = create_engine('sqlite:///'+ database_filename)

    # test if there is a table already there, if so delete then save the new one
    inspector = inspect(engine)

    if len(inspector.get_table_names())>0:
        metadata = MetaData()
        some_table = Table("messages", metadata, autoload_with=engine)
        conn = engine.connect()
        conn.execute(DropTable(some_table))

    df.to_sql('messages', engine, index=False)
    print("successfull saving !!!")
        
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()