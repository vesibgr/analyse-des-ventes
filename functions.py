def isPrimaryKey(df, columnList):
    """
    returns True if the set of columns is a primary key of the DataFrame
    
    args
        df: The DataFrame to test
        columnList: a list of columns
        
    returns
        boolean
    """
    # we test if each columns given exist in df
    for columnInput in columnList:
        if columnInput not in df.columns:
            raise ValueError("'{}' is not a valid column".format(columnInput))
    
    # --> is there two identic lines when we project df ?
    # we project df into the given columns
    # we delete the duplicates of the projection
    # we count the number of line and compare it to the initial one
    return len(df) == len(df.drop_duplicates(subset=columnList))