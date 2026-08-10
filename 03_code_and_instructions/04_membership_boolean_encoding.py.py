

'''
'''
import pandas as pd

def set_levelpoints(level_tuple: tuple):  # (100.608,492.92,721.736)
    level_010 = []
    level_101 = []
    level_010.append((level_tuple[0], 0))
    level_010.append((level_tuple[1], 1))
    level_010.append((level_tuple[2], 0))

    level_101.append((level_tuple[0], 1))
    level_101.append((level_tuple[1], 0))
    level_101.append((level_tuple[2], 1))

    return level_010, level_101


def get_value_010(level_points: list, input_value):
    low_level = level_points[0]
    mid_level = level_points[1]
    high_level = level_points[2]

    x1, y1 = low_level[0], low_level[1]
    x2, y2 = mid_level[0], mid_level[1]
    x3, y3 = high_level[0], high_level[1]

    if input_value <= x1:
        return (0, 0, 0)  

    if input_value >= x3:
        return (0, 0, 0)

    if input_value >= x1 and input_value <= x2:
        res = (input_value - x1) / (x2 - x1) * (y2 - y1) + y1  
        return (0, round(res, 3), 0)

    if input_value >= x2 and input_value <= x3:
        res = (input_value - x2) / (x3 - x2) * (y3 - y2) + y2  
        return (0, 0, round(res, 3))


def get_value_101(level_points: list, input_value):
    low_level = level_points[0]
    mid_level = level_points[1]
    high_level = level_points[2]

    x1, y1 = low_level[0], low_level[1]
    x2, y2 = mid_level[0], mid_level[1]
    x3, y3 = high_level[0], high_level[1]

    if input_value <= x1:
        return (1, 0, 0)  

    if input_value >= x3:
        return (0, 0, 1)

    if input_value >= x1 and input_value <= x2:
        res = (input_value - x1) / (x2 - x1) * (y2 - y1) + y1  
        return (round(res, 3), 0, 0)

    if input_value >= x2 and input_value <= x3:
        res = (input_value - x2) / (x3 - x2) * (y3 - y2) + y2  
        return (0, round(res, 3), 0)


def add_tuples(tuple1, tuple2):
    added_tuple = tuple(x + y for x, y in zip(tuple1, tuple2))
    return added_tuple


# level_points = [(100.608, 0), (492.92, 1),(721.736, 0)]
# level_points2 = [(100.608, 1), (492.92, 0),(721.736, 1)]
# input_value = 108

# output_value = get_value_010(level_points, input_value)
# print(output_value)
# output_value = get_value_101(level_points2, input_value)
# print(output_value)

def main_func(input_value, datas: tuple):
    level_010, level_101 = set_levelpoints(datas)
    finalRes = add_tuples(get_value_010(level_010, input_value), get_value_101(level_101, input_value))
    print(input_value,'====>',finalRes)
    return finalRes

if __name__ == '__main__':
    # main_func(108, (100.608, 492.92, 721.736))
    # main_func(28.44, (21.023, 30.261, 34.604))

    df = pd.read_excel('数据.xlsx',index_col=None)
    for i in (df['N6']):
        main_func(i, (459.622473372781, 3313.965862626, 8151.9776))