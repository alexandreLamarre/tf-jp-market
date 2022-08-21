import logging
import pathlib
from dataclasses import dataclass
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class CSVReader:
	read_path: str
	logger: logging.Logger = logging.getLogger(__name__)
	data: bytes = None
	df: pd.DataFrame = None
	
	def to_raw(self):
		self.data = pathlib.Path(self.read_path).read_bytes()
	
	def to_pandas(self):
		self.df = pd.read_csv(self.read_path)
	
	def train_linear_regression(self, attr1: str, attr2: str) -> LinearRegression:
		X = self.df.loc[:, [attr1]]
		y = self.df.loc[:, attr2]
		model = LinearRegression()
		model.fit(X, y)
		return model



def main():
	csv_oil = CSVReader(read_path='data/oil.csv')
	csv_oil.to_pandas()
	csv_stores = CSVReader(read_path='data/stores.csv')
	csv_stores.to_pandas()
	csv_holidays = CSVReader(read_path='data/holidays_events.csv')
	csv_holidays.to_pandas()
	csv_stores = CSVReader(read_path='data/stores.csv')
	csv_stores.to_pandas()
	csv_transactions = CSVReader(read_path='data/transactions.csv')
	csv_transactions.to_pandas()


if __name__ == "__main__":
	main()
