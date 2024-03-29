import numpy as np
import pandas as pd
DD = np.array([ [0    ,   963   ,  429  ,   1949,    2979   , 1504  ,  206  ,   2976  ,  3095],
                [963 ,    0   ,    671    , 996    , 2054   , 1329 ,   802   ,  2013  ,  2142],
                [429  ,   671  ,   0    ,  1616   , 2631   , 1075   , 233  ,   2684   , 2799],
                [1949   , 996   ,  1616  ,  0  ,     1059  ,  2037   , 1771   , 1307    ,1235],
                [2979   , 2054 ,   2631 ,   1059  ,  0    ,   2687  ,  2786  ,  1131  ,  379],
                [1504   , 1329  ,  1075 ,   2037  ,  2687 ,   0     ,  1308  ,  3273  ,  3053],
                [206   ,  802    , 233  ,   1771  ,  2786 ,   1308  ,  0     ,  2815  ,  2934],
                [2976   , 2013    ,2684 ,   1307  ,  1131 ,   3273  ,  2815  ,  0     ,  808],
                [3095    ,2142   , 2799 ,   1235  ,  379  ,   3053  ,  2934  ,  808   ,  0] ])

ddf = pd.DataFrame(DD,columns=['BOS'  ,   'CHI'  ,   'DC'     , 'DEN'   ,  'LA'    ,  'MIA'    , 'NY'  ,    'SEA'  ,  'SF'])