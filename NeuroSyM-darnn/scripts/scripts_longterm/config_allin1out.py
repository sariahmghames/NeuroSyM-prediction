from typing import List, Optional
from pathlib import Path

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json # dataclasses_json is an api (so .api)
from cached_property import cached_property

import numpy as np


@dataclass_json 
@dataclass 

class Config(object):
    """
    Attributes
    ----------

    decay_rate:
        Decay rate of the momentum

    data_paths:
        List of paths to input files. The input files can be split into
        multiple csv as long as they have same header and format.
        When loading the data through `get_train_test_datasets` one can
        choose whether the csv have to be merged (when loaded) and then
        transformed into windows or merged only after windowing. This
        was done since the SML2010 dataset is split into two csvs with the
        second starting at a timestep far later than the first one


    target_cols:
        The name of the columns that are the target values (y) for the
        prediction. When the dataset is returned from
        `get_train_test_datasets` the X will contain values of the columns in
        (header minus (target_cols union drop_cols)) while the y values of the
        `target_cols`.

    drop_cols:
        The name of the columns to be dropped from the csv.
        In the case of SML2010 some columns are 0, such as the ones regarding
        Exterior_Entalpic_*

    n:
        Number of driving series

    m:
        The size of the hidden state of the encoder. In the case of encoder
        decoder model this will be also the decoder's hidden size

    p:
        The size of the hidden state of the decoder. Ignored in encoder
        decoder model

    sep:
        The pattern separating columns in the csvs

    T:
        The number of past values the predictor can perceive. T values
        of the driving series (X) and T-1 values for the target series.
        The T-th values of the target series (y) are the ones to be predicted

    learning_rate:
        Learning rate for optimizing the parameters ................................ rate at which to update the parameters

    decay_frequency:

    max_gradient_norm:
        Used in gradient clipping

    optimizer:
        Optimizer name

    batch_size:
        Number of windows to be processed at the same time

    num_epochs:
        Number of epochs to train the network

    log_dir:
        Directory for logging

    train_ratio:
        Portion of the data to be used as training set. The remainder of
        the portion is equally split into test and validation.

    report_frequency:
        Print loss and train speed each [this param] batches

    plot_frequency:
        Plot true and predicted curves each [this param] epochs

    seed:
        Seed used by frameworks to ensure reproducibility

    inp_att_enabled:
        Whether the input attention is enabled or not
    temporal_att_enabled:
        Whether the temporal attention is enabled or not
    """

    decay_rate: float
    data_paths: List[str]
    target_cols: List[str]
    drop_cols: Optional[List[str]] = field(default_factory=list)
    n: int = 10
    qtc: int = 6
    m: int = 64
    p: int = 64
    sep: str = ","
    T: int = 5  
    features: int = 2
    enc: int = 100
    classes: int = 1
    labels: int = 1
    nb_steps_ahead: int = 1
    learning_rate: float = 0.001          
    decay_frequency: int = 1000
    max_gradient_norm: float = 5
    optimizer: str = "adam"
    batch_size: int = 128
    num_epochs: int = 10
    log_dir: str = "log/"
    train_ratio: float = 0.8
    report_frequency: int = 50
    plot_frequency: int = 5  
    seed: int = 42
    inp_att_enabled: bool = True
    temporal_att_enabled: bool = True
    update_alpha: bool = False
    dropout_rate: float = 0.5


    @cached_property
    def log_path(self):
        return Path(self.log_dir)


    @cached_property
    def usecols_driving(self):
        cols_driving = []
        for i in range(0,int(len(self.data_paths)/3)):
            path = self.data_paths[i]
            with open(path) as f:
                header = f.readline().strip().split(self.sep)
                cols1 = [col for col in header if col not in self.drop_cols]
                #print("cols1 in shape=", len(cols1)) # uncomment for debugging dataset storage
                cols_driving.append([col for col in header if col not in self.drop_cols])
        return cols_driving



    @cached_property
    def usecols_target(self):
        cols_target = []
        for i1 in range(int(len(self.data_paths)/3), 2*int(len(self.data_paths)/3)):
            path = self.data_paths[i1]        
            with open(path) as f1:
                header1 = f1.readline().strip().split(self.sep)
                cols2 = [col for col in header1 if col not in self.drop_cols]
                #print("cols2 in shape=", len(cols2))
                cols_target.append([col1 for col1 in header1 if col1 not in self.drop_cols])
        return cols_target




    @cached_property
    def usecols_alpha(self):
        cols_alpha = []
        for i2 in range(2*int((len(self.data_paths))/3), int(len(self.data_paths)), 1):
            path = self.data_paths[i2]   
            with open(path) as f2:
                header2 = f2.readline().strip().split(self.sep)
                cols3 = [col for col in header2 if col not in self.drop_cols]
                #print("cols3 in shape=", len(cols3))
                cols_alpha.append([col2 for col2 in header2 if col2 not in self.drop_cols])
        return cols_alpha





    @cached_property
    def driving_series(self):
        driving_data = []
        driving_cols = self.usecols_driving
        for one_cluster in driving_cols:
            driving_data.append([col for col in one_cluster if col not in self.target_cols])
        return driving_data



    @cached_property
    def target_series(self):
        target_data = []
        target_cols = self.usecols_target
        for one_cluster in target_cols:
            target_data.append([col for col in one_cluster if col not in self.target_cols])
        return target_data



    @cached_property
    def alpha_series(self):
        alpha_data = []
        alpha_cols = self.usecols_alpha
        for one_cluster in alpha_cols:
            alpha_data.append([col for col in one_cluster if col not in self.target_cols])
        return alpha_data


    @cached_property
    def n(self):
        return len(self.driving_series)  


    @classmethod
    def from_file(cls, path): 
        with open(path) as f:
            c = cls.from_json(f.read())
        c.log_dir = Path(c.log_dir) / Path(path).with_suffix('').name
        return c



    def to_file(self, path):
        with open(path, "w") as f:
            f.write(self.to_json(indent=4))


# Test
if __name__ == "__main__":
    with open("../conf/JackRabbot_x1_longterm.json") as f:
        c = Config.from_json(f.read())
    print(c.to_json())
