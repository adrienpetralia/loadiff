import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter


class TabularData:

    def __init__(self, list_df):
        self.list_df = list_df

    def presence_list(self):
        if self.list_df is None:
            raise ValueError("Pas de df")
        else:
            pass
