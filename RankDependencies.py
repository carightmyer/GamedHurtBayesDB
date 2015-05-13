import pickle
import copy
import sys
import scipy.sparse.csgraph as csgraph
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#don't touch!
columnsKey = 'column_names'
matrixKey = 'matrix'

#config me!
do_sort = False
apply_threshold = True
do_cc = True
threshold = 0.9
filename = "../s10b-cc-float-0"
target = "ghpred"

def DeserializeData():
    """ Loads the column and dependency data from file
    for us to work with """

    #open target file as read only
    print " -> Opening file " + filename + " with read permissions"
    f = open(filename, 'r')

    #use pickle to convert file to a Python dict
    print " -> De-serializing data frame..."
    pandas_df = pickle.load(f)

    #close the file
    f.close()
    print " -> File " + filename + " closed"

    #extract arrays we need from dict
    columns = None
    data = None
    try:
        columns = pandas_df[columnsKey].tolist()
        data = pandas_df[matrixKey].tolist()
    except AttributeError:
        columns = pandas_df[columnsKey]
        data = pandas_df[matrixKey]

    #if the column length doesn't match the length of the
    #dependency data, we're in trouble
    print " -> Validating data format"
    assert len(columns) == len(data)

    #return the data and columns
    return [data, columns]

def SortData(data, columns, target_column):
    """ Sorts the given data by the specified column, descending"""

    #single row, representing dependencies between our target
    #column and the rest of the data. the rest of the data will
    #be sorted by this relationship.
    target_vals = []

    #we will use these indices to sort additional rows
    indices = []

    #this will repesent the whole collection of sorted data
    #and will overwrite the data argument
    sorted_vals = []

    #simply the index of the column in question
    tci = columns.index(target)

    #get the dependency values for only target column
    for i in range(0, len(columns)):
        target_vals.append([columns[i], data[i][tci], i])

    #determine order by descending dependency
    target_vals = sorted(target_vals, key=lambda x: x[1], reverse=True)
    print " -> Sorting values by dependence to column " + target

    #determine indices by sorted dependency results
    for i in range(0, len(columns)):
        indices.append(target_vals[i][2])

    #re-sort column order accordingly
    new_columns = []
    for i in range(0, len(columns)):
        new_columns.append(target_vals[i][0])

    #loop through rows and figure out the new
    #index for each of the rows and columns
    for i in range(0, len(columns)):
        i_index = target_vals[i][2]
        d = data[i_index]
        d_sorted = []
        for j in range(0, len(columns)):
            ji = target_vals[j][2]
            d_sorted.append(d[ji])
        sorted_vals.append(d_sorted)

    #overwrite old data (or we could return a tuple)
    data = sorted_vals
    columns = new_columns
    return [data, columns]

def CalculateTypicality(data, columns):
    """ or typicality AKA I don't really know what this is """

    #just sum up and print the sums of all columns?
    for i in range(0, len(columns) - 1):
        column_sum = 0.0
        for j in range(0, len(columns) - 1):
            column_sum += data[i][j]
        column_sum = column_sum / len(columns)
        print " -> sum of: " + str(new_columns[i]) + " = " + str(column_sum)

def ApplyThreshold(data, threshold):
    """ Applies a threshold to round data points to 0.0 or 1.0 """

    #loop through each data point and floor or ceiling it
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            if data[i][j] < threshold:
                data[i][j] = 0
            else:
                data[i][j] = 1

def SaveToImage(data, columns, fn):
    """ Save the given data to a labeled plot """

    #set up matplotlib to plot our data
    fig, ax = plt.subplots()
    cm = plt.cm.winter_r
    cbp = ax.imshow(data, cmap=cm, interpolation='nearest')

    #create x/y labels
    plt.yticks(np.arange(len(columns)), columns)
    plt.xticks(np.arange(len(columns)), columns, rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.24)

    #color scale
    cbar = plt.colorbar(cbp)
    #cbar.ax.set_yticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'])
    #cbar.set_label('pairwise dependence probability', rotation=270)

    #save the figure
    print " -> Saving figure to " + fn
    plt.savefig(fn)

def ConnectedComponents(data, columns, originalFloats):
    cc = csgraph.connected_components(data)[1]
    groups = []
    group = []
    prev = -1
    for i in range(0, len(data)):
        if i == len(data) - 1 and len(group) > 1:
            groups.append(group)
        if cc[i] == prev:
            group += [columns[i]]
        else:
            if len(group) > 1:
                groups.append(group)
            group = [columns[i]]
            prev = cc[i]

    #overview of connected components
    print " -> Connected Components: " + str(groups)
    for i in range(0, len(groups)):
        print " || -> Block " + str(i) + ": " + str(groups[i])

    #perform a deeper analysis on every group of connected components
    for tbi in range(0, len(groups)):

        #reduce the graph to the major connected components
        blob = {}
        newData = []
        newFloats = []
        for i in range(0, len(groups[tbi])):
            row = data[columns.index(groups[tbi][i])]
            frow = originalFloats[columns.index(groups[tbi][i])]
            trimmed = []
            floats = []
            for j in range(0, len(groups[tbi])):
                trimmed.append(row[columns.index(groups[tbi][j])])
                floats.append(frow[columns.index(groups[tbi][j])])
            newData.append(trimmed)
            newFloats.append(floats)

        #determine the most typical component in each block
        print " -> Major components analysis for block:" + str(tbi)
        bestSum = 0.0
        bestColumnIndex = 0
        for i in range(0, len(newFloats)):
            column_sum = 0.0
            for j in range(0, len(newFloats)):
                column_sum += newFloats[i][j]
            print " || -> Sum dependence for " + groups[tbi][i] + ": " + str(column_sum)

        #save this reduced graph to file
        blob[columnsKey] = groups[tbi]
        blob[matrixKey] = newData
        fn = filename + "-cc-bool-" + str(tbi)
        f = open(fn, 'w')
        pickle.dump(blob, f)
        f.close()
        SaveToImage(newData, groups[tbi], fn + ".png")

        #save reduced float matrix to file
        blob[matrixKey] = newFloats
        fn = filename + "-cc-float-" + str(tbi)
        f = open(fn, 'w')
        pickle.dump(blob, f)
        f.close()
        SaveToImage(newFloats, groups[tbi], fn + ".png")

################################
# configure this to your needs #
################################

#__main__
content = DeserializeData()
data = content[0]
columns = content[1]
floats = copy.deepcopy(data)

if do_sort:
    data, columns = SortData(data, columns, target)

if apply_threshold:
    ApplyThreshold(data, threshold)

if do_cc:
    ConnectedComponents(data, columns, floats)

outfile = filename + "_t-" + str(threshold) + ".png"
SaveToImage(data, columns, outfile)
