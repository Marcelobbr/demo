# import json
# import numpy as np
# import pandas as pd
# import os
# import sys
# from datetime import datetime

class ProjectParameters:

    def __init__(self):
        self.target_type = 'binary'
        if self.target_type == 'regression':
            self.scoring = 'neg_mean_squared_error'
        elif self.target_type == 'binary':
            self.scoring = 'accuracy'

        self.numerical_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

