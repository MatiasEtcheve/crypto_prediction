import pandas as pd
import vectorbt as vbt
import numpy as np


def apply_logical_operation(df1, df2, operator):
    new_index = vbt.base.index_fns.combine_indexes((df1.columns, df2.columns))
    duplicate_index = [
        i
        for i in range(len(new_index.names))
        if new_index.names[i] in new_index.names[:i]
    ]
    new_index = vbt.base.index_fns.drop_levels(
        new_index, duplicate_index
    ).drop_duplicates()
    df = pd.DataFrame("_", df1.index, new_index)

    def combine_func(*args, first_columns, second_columns):
        name = args[0].name
        first_col_index = np.array(name, dtype="O")[first_columns]
        second_col_index = np.array(name, dtype="O")[second_columns]

        if operator == "&":
            return (
                df1.loc[:, tuple(first_col_index)] & df2.loc[:, tuple(second_col_index)]
            )

        if operator == "|":
            return (
                df1.loc[:, tuple(first_col_index)] | df2.loc[:, tuple(second_col_index)]
            )

        raise ValueError(f"bruh what is that operation: {operator}")

    column_to_take_in_first = {
        index: df1.columns.names.index(col_name)
        for index, col_name in enumerate(df.columns.names)
        if col_name in df1.columns.names
    }
    column_to_take_in_second = {
        index: df2.columns.names.index(col_name)
        for index, col_name in enumerate(df.columns.names)
        if col_name in df2.columns.names
    }
    column_in_first = sorted(
        column_to_take_in_first, key=lambda k: column_to_take_in_first[k]
    )
    column_in_second = sorted(
        column_to_take_in_second, key=lambda k: column_to_take_in_second[k]
    )

    return df.apply(
        func=combine_func,
        axis=0,
        first_columns=column_in_first,
        second_columns=column_in_second,
    )
