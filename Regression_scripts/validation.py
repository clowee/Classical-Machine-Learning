import pandas
import numpy

def split_group_cv(df, splits):
    tr = []
    te = []
    for i in range(1, splits + 1):
        te.append(df.index[df["Groups"] == i + 1].values.tolist())
        tr.append(df.index[(df["Groups"] < i + 1) & (df["Groups"] < splits + 1)].values.tolist())
    split = zip(tr, te)
    return split

def create_groups_cv(raw_df, splits):
    projects = raw_df.project.unique().tolist()
    # create empty "group" column where to write number of group
    raw_df["Groups"] = ""
    # define the number of folds
    folds = splits
    # select by project - iterate through all projects TODO run in parallel for all projects (useful for bigger datasets)
    allProjects_df = []
    # create a df for each project
    for i in projects:
        project_df = raw_df.loc[raw_df["project"] == i]
        project_df = project_df.reset_index(drop=True)
        otherGroups, remainder = divmod(len(project_df), (
                folds + 1))  # divides lenght of the project and find the result (to be used for groups 2 to k, and the remainder to be added to the group 1
        firstGroup = otherGroups + remainder

        # create list of position for each group (to be used when writing the the group number)
        groupsDim = [0, firstGroup, firstGroup + otherGroups]
        for j in range(2, folds + 1):
            groupsDim.append(otherGroups + groupsDim[j])

        # add group number to set of rows in the project
        for k in range(folds + 1):
            project_df.loc[groupsDim[k]:groupsDim[k + 1], "Groups"] = k + 1

        allProjects_df.append(project_df)
    # concatenate all individual dataframes created into the final one
    df = pandas.concat(allProjects_df).reset_index(drop=True)

    return df