import pandas as pd
import os 
from mlProject import logger
from sklearn.linear_model import ElasticNet
import joblib
from mlProject.config.configuration import ConfigurationManager
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_model(self):
        logger.info("Loading training and testing data")
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        logger.info("Splitting data into features and target")
        X_train = train_data.drop(self.config.target_column,axis=1)
        y_train = train_data[self.config.target_column]

        X_test = test_data.drop(self.config.target_column,axis=1)
        y_test = test_data[self.config.target_column]

        logger.info("Training the ElasticNet model")
        model = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        model.fit(X_train, y_train)

        logger.info("Saving the trained model")
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))