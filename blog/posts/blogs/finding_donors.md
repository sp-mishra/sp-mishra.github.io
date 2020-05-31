---
title: "Finding Donors"
tags: ["python", "machinelearning", "math", "games"]
published: true
date: "2019-03-02"
---

# Introduction

## Project: Finding Donors for _CharityML_

Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

> **Note:** Please specify WHICH VERSION OF PYTHON you are using when submitting this notebook. Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

## Getting Started

In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than \$50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations. Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with. While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features.

The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

---

## Exploring the Data

Run the code cell below to load necessary Python libraries and load the census data. Note that the last column from this dataset, `'income'`, will be our target label (whether an individual makes more than, or at most, \$50,000 annually). All other columns are features about each individual in the census database.

```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display, HTML # Allows the use of display() for DataFrames
import seaborn as sns
# Import supplementary visualization code visuals.py
import visuals as vs
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.warnings.filterwarnings('ignore')

# Pretty display for notebooks
%matplotlib inline
%lsmagic
# %pylab inline
sns.set(color_codes=True)
# Load the Census dataset
data = pd.read_csv("census.csv")
print('Data upload done')
```

    Data upload done

```python
# Section for helper functions
def display_html(display_string: str, color: str='blue', heading_value: str = 'h4'):
    """
    Display some text in HTML form inside the display sections
    """
    display(HTML(' <span style="color:{0}"><{2}>{1}</{2}> </span>  '.format(color, display_string, heading_value)))

def print_general_information(title:str, data):
    """
    Display general information related to data
    """
    display_html('Display basic information about the records "{}":'.format(title))
    # Success - Display the first record
    display(data.head(n=5))
    display(data.info())
    print("Get an overview of the data before we proceed:")
    display(data.describe())
    display_html('Checking the shape of the data:')
    display(data.shape)
    # Print unique values in the income column. This is the column that will be used for training and prediction.
    display_html('Check the unique values for income:')
    display(data['income'].unique())

def display_pair_plot(title:str, data):
    """
    Display pairplot
    """
    display_html('Pairplot "{}":'.format(title))
    sns.pairplot(data)
    plt.show()
```

```python
print_general_information('Data Variable', data)
```

<span style="color:blue"><h4>Display basic information about the records "Data Variable":</h4> </span>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education_level</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>13.0</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>13.0</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>9.0</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>11th</td>
      <td>7.0</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>Bachelors</td>
      <td>13.0</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 45222 entries, 0 to 45221
    Data columns (total 14 columns):
    age                45222 non-null int64
    workclass          45222 non-null object
    education_level    45222 non-null object
    education-num      45222 non-null float64
    marital-status     45222 non-null object
    occupation         45222 non-null object
    relationship       45222 non-null object
    race               45222 non-null object
    sex                45222 non-null object
    capital-gain       45222 non-null float64
    capital-loss       45222 non-null float64
    hours-per-week     45222 non-null float64
    native-country     45222 non-null object
    income             45222 non-null object
    dtypes: float64(4), int64(1), object(9)
    memory usage: 4.8+ MB



    None


    Get an overview of the data before we proceed:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.547941</td>
      <td>10.118460</td>
      <td>1101.430344</td>
      <td>88.595418</td>
      <td>40.938017</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.217870</td>
      <td>2.552881</td>
      <td>7506.430084</td>
      <td>404.956092</td>
      <td>12.007508</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>47.000000</td>
      <td>13.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>16.000000</td>
      <td>99999.000000</td>
      <td>4356.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
</div>

<span style="color:blue"><h4>Checking the shape of the data:</h4> </span>

    (45222, 14)

<span style="color:blue"><h4>Check the unique values for income:</h4> </span>

    array(['<=50K', '>50K'], dtype=object)

**Checking the pairplot**

```python
display_pair_plot('Data Variable', data)
```

<span style="color:blue"><h4>Pairplot "Data Variable":</h4> </span>

![png](../../src/images/output_8_1.png)

### Implementation: Data Exploration

A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals making more than \$50,000. In the code cell below, you will need to compute the following:

- The total number of records, `'n_records'`
- The number of individuals making more than \$50,000 annually, `'n_greater_50k'`.
- The number of individuals making at most \$50,000 annually, `'n_at_most_50k'`.
- The percentage of individuals making more than \$50,000 annually, `'greater_percent'`.

**HINT:** You may need to look at the table above to understand how the `'income'` entries are formatted.

```python
# TODO: Total number of records
n_records = np.alen(data)

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = data[data.income == '>50K'].shape[0]

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = np.alen(data[data.income == '<=50K'])

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = np.multiply(np.divide(np.float64(n_greater_50k), n_records), 100)

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
```

    Total number of records: 45222
    Individuals making more than $50,000: 11208
    Individuals making at most $50,000: 34014
    Percentage of individuals making more than $50,000: 24.78439697492371%

**Features**

- `age`: Age (continuous)
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed (continuous)
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains (continuous)
- `capital-loss`: Monetary Capital Losses (continuous)
- `hours-per-week`: Average Hours Per Week Worked (continuous)
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**

- `income`: Income Class (<=50K, >50K)

---

## Preparing the Data

Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.

### Transforming Skewed Continuous Features

A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number. Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`.

Run the code cell below to plot a histogram of these two features. Note the range of the values present and how they are distributed.

```python
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)
```

![png](output_14_0.png)

For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.

Run the code cell below to perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed.

```python
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)
display(features_log_transformed.head())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education_level</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>13.0</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>7.684784</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>13.0</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>9.0</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>11th</td>
      <td>7.0</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>Bachelors</td>
      <td>13.0</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>Cuba</td>
    </tr>
  </tbody>
</table>
</div>

![png](output_16_1.png)

### Normalizing Numerical Features

In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as `'capital-gain'` or `'capital-loss'` above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.

Run the code cell below to normalize each numerical feature. We will use [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for this.

```python
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))
vs.distribution(features_log_transformed, transformed = True)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education_level</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.301370</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0.667492</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.452055</td>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.122449</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.287671</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>0.533333</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.493151</td>
      <td>Private</td>
      <td>11th</td>
      <td>0.400000</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.150685</td>
      <td>Private</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>Cuba</td>
    </tr>
  </tbody>
</table>
</div>

![png](output_18_1.png)

```python
# Print the experimental data as pairplots after similar processing
# Data for some further experiment
experiment_data = pd.read_csv("census.csv")
experiment_data.head()
def plot_experimental_pairplot(expdata):
    # Log-transform the skewed features
    skewed_cols = ['capital-gain', 'capital-loss']
    expdata['income'] = expdata['income'].apply(lambda x: 1 if x == '>50K' else 0)
    expdata[skewed_cols] = expdata[skewed_cols].apply(lambda x: np.log(x + 1))
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    expdata[numerical_cols] = scaler.fit_transform(expdata[numerical_cols])
    display(expdata.head(n = 5))
    display_pair_plot('Exp Data Variable', expdata)

plot_experimental_pairplot(experiment_data)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education_level</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.301370</td>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0.667492</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.452055</td>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.122449</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.287671</td>
      <td>Private</td>
      <td>HS-grad</td>
      <td>0.533333</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.493151</td>
      <td>Private</td>
      <td>11th</td>
      <td>0.400000</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.150685</td>
      <td>Private</td>
      <td>Bachelors</td>
      <td>0.800000</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>Cuba</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

<span style="color:blue"><h4>Pairplot "Exp Data Variable":</h4> </span>

![png](output_19_2.png)

### Implementation: Data Preprocessing

From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called _categorical variables_) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.

| #   | someFeature |                            | someFeature_A | someFeature_B | someFeature_C |
| --- | ----------- | -------------------------- | ------------- | ------------- | ------------- |
| 0   | B           |                            | 0             | 1             | 0             |
| 1   | C           | ----> one-hot encode ----> | 0             | 0             | 1             |
| 2   | A           |                            | 1             | 0             | 0             |

Additionally, as with the non-numeric features, we need to convert the non-numeric target label, `'income'` to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("<=50K" and ">50K"), we can avoid using one-hot encoding and simply encode these two categories as `0` and `1`, respectively. In code cell below, you will need to implement the following:

- Use [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) to perform one-hot encoding on the `'features_log_minmax_transform'` data.
- Convert the target label `'income_raw'` to numerical entries.
  - Set records with "<=50K" to `0` and records with ">50K" to `1`.

```python
# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.apply(lambda x: 1 if x == '>50K' else 0)

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

display(features_final.head())
features_final.describe()
# display_html("Encoded Value:")
# display(encoded)
```

    103 total features after one-hot encoding.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>workclass_ Federal-gov</th>
      <th>workclass_ Local-gov</th>
      <th>workclass_ Private</th>
      <th>workclass_ Self-emp-inc</th>
      <th>workclass_ Self-emp-not-inc</th>
      <th>...</th>
      <th>native-country_ Portugal</th>
      <th>native-country_ Puerto-Rico</th>
      <th>native-country_ Scotland</th>
      <th>native-country_ South</th>
      <th>native-country_ Taiwan</th>
      <th>native-country_ Thailand</th>
      <th>native-country_ Trinadad&amp;Tobago</th>
      <th>native-country_ United-States</th>
      <th>native-country_ Vietnam</th>
      <th>native-country_ Yugoslavia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.301370</td>
      <td>0.800000</td>
      <td>0.667492</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.452055</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.122449</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.287671</td>
      <td>0.533333</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.493151</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.150685</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.397959</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 103 columns</p>
</div>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>workclass_ Federal-gov</th>
      <th>workclass_ Local-gov</th>
      <th>workclass_ Private</th>
      <th>workclass_ Self-emp-inc</th>
      <th>workclass_ Self-emp-not-inc</th>
      <th>...</th>
      <th>native-country_ Portugal</th>
      <th>native-country_ Puerto-Rico</th>
      <th>native-country_ Scotland</th>
      <th>native-country_ South</th>
      <th>native-country_ Taiwan</th>
      <th>native-country_ Thailand</th>
      <th>native-country_ Trinadad&amp;Tobago</th>
      <th>native-country_ United-States</th>
      <th>native-country_ Vietnam</th>
      <th>native-country_ Yugoslavia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>...</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
      <td>45222.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.295177</td>
      <td>0.607897</td>
      <td>0.064342</td>
      <td>0.042423</td>
      <td>0.407531</td>
      <td>0.031091</td>
      <td>0.068551</td>
      <td>0.736522</td>
      <td>0.036398</td>
      <td>0.083941</td>
      <td>...</td>
      <td>0.001371</td>
      <td>0.003870</td>
      <td>0.000442</td>
      <td>0.002233</td>
      <td>0.001216</td>
      <td>0.000641</td>
      <td>0.000575</td>
      <td>0.913095</td>
      <td>0.001835</td>
      <td>0.000509</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.181067</td>
      <td>0.170192</td>
      <td>0.214240</td>
      <td>0.190454</td>
      <td>0.122526</td>
      <td>0.173566</td>
      <td>0.252691</td>
      <td>0.440524</td>
      <td>0.187281</td>
      <td>0.277303</td>
      <td>...</td>
      <td>0.037002</td>
      <td>0.062088</td>
      <td>0.021026</td>
      <td>0.047207</td>
      <td>0.034854</td>
      <td>0.025316</td>
      <td>0.023971</td>
      <td>0.281698</td>
      <td>0.042803</td>
      <td>0.022547</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.150685</td>
      <td>0.533333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.397959</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.273973</td>
      <td>0.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.397959</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.410959</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.448980</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 103 columns</p>
</div>

### Shuffle and Split Data

Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.

Run the code cell below to perform this split.

```python
# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    test_size = 0.2,
                                                    random_state = 43)
# I use 43 as seed mostly because it introduced me to the Additive number theory in last one month
# 43 is the smallest prime number expressible as the sum of 2, 3, 4, or 5

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
```

    Training set has 36177 samples.
    Testing set has 9045 samples.

---

## Evaluating Model Performance

In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of your choice, and the fourth algorithm is known as a _naive predictor_.

### Metrics and the Naive Predictor

_CharityML_, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, _CharityML_ is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that _does not_ make more than \$50,000 as someone who does would be detrimental to _CharityML_, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \$50,000 is _more important_ than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:

$$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$

In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).

Looking at the distribution of classes (those who make at most 50,000, and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect **accuracy**, since we could simply say _"this person does not make more than \$50,000"_ and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the _naive prediction_ for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, _CharityML_ would identify no one as donors.

#### Note: Recap of accuracy, precision, recall

**Accuracy** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).

**Precision** tells us what proportion of messages we classified as spam, actually were spam.
It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classificatio), in other words it is the ratio of

`[True Positives/(True Positives + False Positives)]`

**Recall(sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam.
It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of

`[True Positives/(True Positives + False Negatives)]`

For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average(harmonic mean) of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score(we take the harmonic mean as we are dealing with ratios).

### Question 1 - Naive Predictor Performace

- If we chose a model that always predicted an individual made more than \$50,000, what would that model's accuracy and F-score be on this dataset? You must use the code cell below and assign your results to `'accuracy'` and `'fscore'` to be used later.

**Please note** that the the purpose of generating a naive predictor is simply to show what a base model without any intelligence would look like. In the real world, ideally your base model would be either the results of a previous model or could be based on a research paper upon which you are looking to improve. When there is no benchmark model set, getting a result better than random choice is a place you could start from.

**HINT:**

- When we have a model that always predicts '1' (i.e. the individual makes more than 50k) then our model will have no True Negatives(TN) or False Negatives(FN) as we are not making any negative('0' value) predictions. Therefore our Accuracy in this case becomes the same as our Precision(True Positives/(True Positives + False Positives)) as every prediction that we have made with value '1' that should have '0' becomes a False Positive; therefore our denominator in this case is the total number of records we have in total.
- Our Recall score(True Positives/(True Positives + False Negatives)) in this setting becomes 1 as we have no False Negatives.

```python
'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

# Calculating the performance without sklearn
display_html("Calculating the performance without sklearn:")
TP = np.float64(np.sum(income))
FP = np.float64(np.subtract(income.count(), TP))
TN = np.float64(0.0)
FN = np.float64(0.0)

# TODO: Calculate accuracy, precision and recall
accuracy = np.divide(np.float64(TP), income.count())
precision = np.divide(TP, np.add(TP, FP))
recall = np.divide(TP, np.add(TP, FN))
print('accuracy = {}, recall = {}, precision = {}'.format(accuracy, recall, precision))

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
def calculate_fscore(precision, recall, b):
    b2 = np.float64(np.square(b))
    numerator = np.multiply(precision, recall)
    numerator = np.multiply(numerator, (b2 + 1))
    denominator = np.add(np.multiply(b2, precision), recall)
    fscore_ret = np.divide(numerator, denominator)
    return fscore_ret

beta = 0.5
fscore = calculate_fscore(precision=precision, recall=recall, b=beta)

# Print the results
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

# Calculating the performance with sklearn
display_html("Calculating the performance with sklearn:")
all_one_pred = [np.float64(1) for i in range(income.size)]
accuracy = accuracy_score(income, all_one_pred)
precision = precision_score(y_true=income, y_pred=all_one_pred)
recall = recall_score(y_true=income, y_pred=all_one_pred)
fscore = calculate_fscore(precision=precision, recall=recall, b=beta)
print('accuracy = {}, recall = {}, precision = {}'.format(accuracy, recall, precision))
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
```

<span style="color:blue"><h4>Calculating the performance without sklearn:</h4> </span>

    accuracy = 0.2478439697492371, recall = 1.0, precision = 0.2478439697492371
    Naive Predictor: [Accuracy score: 0.2478, F-score: 0.2917]

<span style="color:blue"><h4>Calculating the performance with sklearn:</h4> </span>

    accuracy = 0.2478439697492371, recall = 1.0, precision = 0.2478439697492371
    Naive Predictor: [Accuracy score: 0.2478, F-score: 0.2917]

### Supervised Learning Models

**The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**

- Gaussian Naive Bayes (GaussianNB)
- Decision Trees
- Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
- K-Nearest Neighbors (KNeighbors)
- Stochastic Gradient Descent Classifier (SGDC)
- Support Vector Machines (SVM)
- Logistic Regression

### Question 2 - Model Application

List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen

- Describe one real-world application in industry where the model can be applied.
- What are the strengths of the model; when does it perform well?
- What are the weaknesses of the model; when does it perform poorly?
- What makes this model a good candidate for the problem, given what you know about the data?

**HINT:**

Structure your answer in the same format as above^, with 4 parts for each of the three models you pick. Please include references with your answer.

**Answer:**

This is a classification problem and the output is either 0 or 1 so following are the methods I will be using to test on the census data (The links for various articles and books are provided in the reference section at the end):

_`Logistic Regression`_

- _Real World Application_: [Investigation of risk factors associated with injuries to horses undertaking jump racing in Great Britain](http://theses.gla.ac.uk/4624/56/2013reardonphd.pdf)
- _Strengths of the model_:

* Easy to underatand and interprete, so a good baseline to start with and quickly get some answer.
* Easy on computing resources and effecient to train
* It has low variance and so is less prone to over-fitting
* Can provide good results in case of less features

- _Weakness of the model_:

* Cannot solve non-linear problem with this, the descision surface is linear for logistic regression.
* Sensitive to outliers.
* Logistic regression requires that the variables are independent, so in the training data care has to be given to include only indepenedent variables.
* Needs large sample size to provide stable results.

- _Why use_:

* This is a binary clasification problem of detecting if the income is over 50K or not. So logistic regression can be employed here. The benefit of having logistic regression is that its simple to implement and then the model can be improved( like using stochastic gradient descent). I believe this will act as a better baseline than the naive method we used earlier.

_`Support Vector Machines (SVM)`_

- _Real World Application_: [Using SVM for natural language processing to find out the study region in environmental science](http://aclweb.org/anthology/U12-1016)
- _Strengths of the model_:

* Effective in high dimensional spaces so can model non linear decision boundaries.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

- _Weakness of the model_:

* SVN works by creating hyper planes on n-dimensional feature space, so for larger feature set training SVM can be time consuming if the dataset is big.
* Overfit problem can occur when the data is too noisy
* If the number of features is much greater than the number of samples it may suffer from overfitting if the kernel is not choosen properly. So choosing the kernel is bit essential.

- _Why use_:

* SVM works effectively in the binary classification that we are having here. Apart from that from the pair plots plotted. I do not see a very clear 2-dimensional distinction descision surface. This will be helpful as it operates on hyperplanes to work with data. Here the other task is to find a best possible boundary, SVM can be useful as its built on large margin classification. This will effectively help classify the margins in a better way.

_`Gradient Boosting`_

- _Real World Application_: [Predict survival for cancer patient](https://www.psiweb.org/docs/default-source/default-document-library/guiyuan-lei-slides.pdf?sfvrsn=2526dedb_0)
- _Strengths of the model_:

* Good on large data sets. Good choice to reduce bias and variance.
* Good for both linear and non linear data set.
* Good for both regression and classification tasks.
* New predictors learn from mistakes committed by previous predictors, so it takes less time/iterations to reach close to actual predictions.

- _Weakness of the model_:

* Very sensitive to feature set and training set.
* Predictions are not easy to understand. This may affect the chance that it will be tuned right in the first shot. May need better understanding.
* If stopping crieteria is not choosen properly can lead to overfitting.

- _Why use_:

* The data we have may not be necessarily linear so gradient boosting can be applied here as it works with both kind of data. The data set looks to have some class imbalance and a ensemble method like Gradient boosting can help achieve a better prediction model.

### Implementation - Creating a Training and Predicting Pipeline

To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
In the code block below, you will need to implement the following:

- Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
- Fit the learner to the sampled training data and record the training time.
- Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
  - Record the total prediction time.
- Calculate the accuracy score for both the training subset and testing set.
- Calculate the F-score for both the training subset and testing set.
  - Make sure that you set the `beta` parameter!

```python
# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[0:sample_size], y_train[0:sample_size])
    end = time() # Get end time

    # TODO: Calculate the training time
    results['train_time'] = end - start

    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time

    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start

    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    beta = 0.5
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta = beta)

    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = beta)

    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    # Return the results
    return results
```

### Implementation: Initial Model Evaluation

In the code cell, you will need to implement the following:

- Import the three supervised learning models you've discussed in the previous section.
- Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
  - Use a `'random_state'` for each model you use, if provided.
  - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
- Calculate the number of records equal to 1%, 10%, and 100% of the training data.
  - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.

**Note:** Depending on which algorithms you chose, the following implementation may take some time to run!

```python
# TODO: Import the three supervised learning models from sklearn
np.warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
# TODO: Initialize the three models
random_state = 43
# Logistic Regression
clf_A = LogisticRegression(random_state=random_state)
# Support Vector Machines (SVM)
clf_B = SVC(random_state=random_state)
# Gradient Boosting
clf_C = GradientBoostingClassifier(random_state=random_state)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = y_train.size
samples_10 = np.int64(np.multiply(samples_100, 0.1))
samples_1 = np.int64(np.multiply(samples_100, 0.01))

# Collect results on the learners
# results = {}
# model_set = [clf_A, clf_B, clf_C]
# model_set = [clf_A, clf_C]
def calc_result_of_learner(model_set):
    results = {}
    for clf in model_set:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = \
            train_predict(clf, samples, X_train, y_train, X_test, y_test)
    return results

def print_learner_results_in_table(results, accuracy, fscore):
    for res in results.items():
        display_html(display_string=res[0], color='blue', heading_value='h5')
        display(pd.DataFrame(res[1]).rename(columns={0:'1%', 1:'10%', 2:'100%'}))
    vs.evaluate(results, accuracy, fscore)

# Run metrics visualization for the three supervised learning models chosen
display_html("Run metrics visualization for the three supervised learning models chosen (With SVM):")
model_set = [clf_A, clf_B, clf_C]
results = calc_result_of_learner(model_set=model_set)
print_learner_results_in_table(results=results, accuracy=accuracy, fscore=fscore)
# vs.evaluate(results, accuracy, fscore)
display_html("Run metrics visualization for the three supervised learning models chosen (Without SVM):")
model_set = [clf_A, clf_C]
results = calc_result_of_learner(model_set=model_set)
# vs.evaluate(results, accuracy, fscore)
print_learner_results_in_table(results=results, accuracy=accuracy, fscore=fscore)
```

<span style="color:blue"><h4>Run metrics visualization for the three supervised learning models chosen (With SVM):</h4> </span>

    LogisticRegression trained on 361 samples.
    LogisticRegression trained on 3617 samples.
    LogisticRegression trained on 36177 samples.
    SVC trained on 361 samples.
    SVC trained on 3617 samples.
    SVC trained on 36177 samples.
    GradientBoostingClassifier trained on 361 samples.
    GradientBoostingClassifier trained on 3617 samples.
    GradientBoostingClassifier trained on 36177 samples.

<span style="color:blue"><h5>LogisticRegression</h5> </span>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.823991</td>
      <td>0.841570</td>
      <td>0.843118</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.860000</td>
      <td>0.850000</td>
      <td>0.830000</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.638170</td>
      <td>0.680017</td>
      <td>0.682812</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.727273</td>
      <td>0.691318</td>
      <td>0.642202</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.007558</td>
      <td>0.004691</td>
      <td>0.002988</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.003045</td>
      <td>0.019732</td>
      <td>0.246644</td>
    </tr>
  </tbody>
</table>
</div>

<span style="color:blue"><h5>SVC</h5> </span>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.758872</td>
      <td>0.834052</td>
      <td>0.841791</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.763333</td>
      <td>0.850000</td>
      <td>0.840000</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.000000</td>
      <td>0.668927</td>
      <td>0.682872</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.000000</td>
      <td>0.703422</td>
      <td>0.670103</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.207695</td>
      <td>1.739599</td>
      <td>14.293886</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.009839</td>
      <td>0.833704</td>
      <td>89.733363</td>
    </tr>
  </tbody>
</table>
</div>

<span style="color:blue"><h5>GradientBoostingClassifier</h5> </span>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.830735</td>
      <td>0.863460</td>
      <td>0.864787</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.966667</td>
      <td>0.890000</td>
      <td>0.876667</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.652447</td>
      <td>0.735864</td>
      <td>0.738580</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.929577</td>
      <td>0.787781</td>
      <td>0.759076</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.018142</td>
      <td>0.016104</td>
      <td>0.019581</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.077291</td>
      <td>0.644665</td>
      <td>7.494051</td>
    </tr>
  </tbody>
</table>
</div>

![png](output_34_8.png)

<span style="color:blue"><h4>Run metrics visualization for the three supervised learning models chosen (Without SVM):</h4> </span>

    LogisticRegression trained on 361 samples.
    LogisticRegression trained on 3617 samples.
    LogisticRegression trained on 36177 samples.
    GradientBoostingClassifier trained on 361 samples.
    GradientBoostingClassifier trained on 3617 samples.
    GradientBoostingClassifier trained on 36177 samples.

<span style="color:blue"><h5>LogisticRegression</h5> </span>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.823991</td>
      <td>0.841570</td>
      <td>0.843118</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.860000</td>
      <td>0.850000</td>
      <td>0.830000</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.638170</td>
      <td>0.680017</td>
      <td>0.682812</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.727273</td>
      <td>0.691318</td>
      <td>0.642202</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.004381</td>
      <td>0.004233</td>
      <td>0.003947</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.002694</td>
      <td>0.017182</td>
      <td>0.212742</td>
    </tr>
  </tbody>
</table>
</div>

<span style="color:blue"><h5>GradientBoostingClassifier</h5> </span>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1%</th>
      <th>10%</th>
      <th>100%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acc_test</th>
      <td>0.830735</td>
      <td>0.863460</td>
      <td>0.864787</td>
    </tr>
    <tr>
      <th>acc_train</th>
      <td>0.966667</td>
      <td>0.890000</td>
      <td>0.876667</td>
    </tr>
    <tr>
      <th>f_test</th>
      <td>0.652447</td>
      <td>0.735864</td>
      <td>0.738580</td>
    </tr>
    <tr>
      <th>f_train</th>
      <td>0.929577</td>
      <td>0.787781</td>
      <td>0.759076</td>
    </tr>
    <tr>
      <th>pred_time</th>
      <td>0.021167</td>
      <td>0.016140</td>
      <td>0.023832</td>
    </tr>
    <tr>
      <th>train_time</th>
      <td>0.082876</td>
      <td>0.605710</td>
      <td>7.868132</td>
    </tr>
  </tbody>
</table>
</div>

![png](output_34_15.png)

---

## Improving Results

In this final section, you will choose from the three supervised learning models the _best_ model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score.

### Question 3 - Choosing the Best Model

- Based on the evaluation you performed earlier, in one to two paragraphs, explain to _CharityML_ which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000.

**HINT:**
Look at the graph at the bottom left from the cell above(the visualization created by `vs.evaluate(results, accuracy, fscore)`) and check the F score for the testing set when 100% of the training set is used. Which model has the highest score? Your answer should include discussion of the:

- metrics - F score on the testing when 100% of the training data is used,
- prediction/training time
- the algorithm's suitability for the data.

**Answer:**

- Of the 3 models tested the `Gradient Boosting Classifier` performed the best. It has scored better in both the testing and training sets. Though its F-Score and accuracy score is nearly same as the other 3 its still on the higher side for both the testing and training data. This means it is a proper balance of prescision (The number of relevent items) and recall (How many relevent items are selected)
- The time taken by the SVM is quite high than the other 3. We can obsereve it by the second set of graph where we are discarding it. Though it takes a significant amout of time more, the accuracy and F-Score on both test and train data are less. So we will be discarding it. Now in Logistic Regression Vs Gradient Boosting; the accuracy and F-Score of the `Gradient Boosting` is bit more for both test and training data. So we can choose it. Apart from that if we see the timing, we can see that with the % increase in samples the time taken to training time for the gradient boosting is quite more, but for the testing set, though with % increase the time to predict for the Logistic regression increases, the time for Gradient Boosting is nearly same. This may be true if the data size increase further.
- We will be using `Gradient Boosting` for further analysis. The algorithm scored better with the default parameters. So improving the hyper parameter tuning may possible give a better predction. Apart from that it is fast to work on.

### Question 4 - Describing the Model in Layman's Terms

- In one to two paragraphs, explain to _CharityML_, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.

**HINT:**

When explaining your model, if using external resources please include all citations.

**Answer:**
The algorith we will be using here is summarised:

1. It first models the data with a simpler model (weak model). Because the model is simple it will not be a good fit to generate errors. Now it focuses on that error.
2. Now it uses a different model (predictor) to fix this hard to fit data (error data) and get them right.
3. The above 2 steps are repeated for some time with different predictors so that we have have better results gradually.
4. At the end we combine the predictors in some way to get better results.

For a simple example lets think we have the task to grade a students paper (lets say math combination of calculus, stats and algebra) and instead of getting expert in the field we stick with some people who are not that exprt but can look at a gradient rubic we are giving them and grade. So the assumption is that individually they will not be doing justice in grading as they will work mostly mechanically. So we follow this to train them:

1. Ask one person to grade with the rubic and we verify the result, point out the errors in the grading process. These errors were due to the fact that this person is good in understanding certain question (lets say calculus) and grading well and not good in grading other questions (algebra and stats).
2. Now we assign the 2nd person with grading the questions where the 1st persion failed and so on repeat the process till we are satisfied.
3. After that we average the results using some method.

This is the gist of the ensemble method like gradient boosting.

Applying the similar example to the problem at hand we have the following solution using gradient boosting:
Here the first simple solution can be lets say all of them have more than 50K salary as we did in the naive method eariler. This may not be accurate as its not a very good prediction but a weak one. The algorithm will check how good it has done, what is the residual (How much it is deviating from the actual results). For next step it will figure out which variable is causing more trouble and then it will use a descision tree to get a better catagorization on that variable. This process is repeated till the satisfied results are reached.

Gradient boosting has benefits for our approach as we have observed that the time taken to train and predict by this mode is really good and also the end results that is given by the F-Score is also good.

### Implementation: Model Tuning

Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:

- Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
- Initialize the classifier you've chosen and store it in `clf`.
- Set a `random_state` if one is available to the same state you set before.
- Create a dictionary of parameters you wish to tune for the chosen model.
- Example: `parameters = {'parameter' : [list of values]}`.
- **Note:** Avoid tuning the `max_features` parameter of your learner if that parameter is available!
- Use `make_scorer` to create an `fbeta_score` scoring object (with $\beta = 0.5$).
- Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
- Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.

**Note:** Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!

```python
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = GradientBoostingClassifier(random_state=random_state, verbose=0)

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
# 3 is the default one. So I am starting from 3
# n_estimators starts with 100
# parameters = {'max_depth':[3,4,5,6,7],
#              'loss':['deviance', 'exponential'],
#              'n_estimators':[100, 150, 300, 600],
#              'learning_rate': [0.1, 0.5, 1.0],
#              'warm_start': [True, False]}

# parameters = {'max_depth':[3,4,5,6,7],
#              'loss':['deviance', 'exponential'],
#              'n_estimators':[100, 150, 300, 600]}

# parameters = {'max_depth':[3,4,5,6,7],
#              'loss':['deviance', 'exponential'],
#              'n_estimators':[100, 150, 300, 600],
#              'learning_rate': [0.1, 0.5, 1.0]}

parameters = {'max_depth':[3,4,5,6,7],
             'loss':['deviance', 'exponential']}

# TODO: Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta = 0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
start = time()
grid_obj = GridSearchCV(estimator = clf, param_grid = parameters, scoring = scorer)


# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)
end = time()
# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
# best_predictions = best_clf.predict(X_test)
```

```python
start = time()
best_predictions = best_clf.predict(X_test)
end = time()
print("Time taken to predict: {}".format(end - start))
```

    Time taken to predict: 0.03802013397216797

```python
# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print('Time taken for grid search: {}s'.format(end-start))
display(best_clf)
```

    Unoptimized model
    ------
    Accuracy score on testing data: 0.8648
    F-score on testing data: 0.7386

    Optimized Model
    ------
    Final accuracy score on the testing data: 0.8724
    Final F-score on the testing data: 0.7527
    Time taken for grid search: 0.03802013397216797s



    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=5,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  n_iter_no_change=None, presort='auto', random_state=43,
                  subsample=1.0, tol=0.0001, validation_fraction=0.1,
                  verbose=0, warm_start=False)

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
conf_mat = confusion_matrix(y_test, best_predictions)
# normalize the data
conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
display(conf_mat)
sns.heatmap(conf_mat, annot=True, annot_kws={"size":50}, cmap='plasma_r', square=False)
plt.title('Confusion matrix for:\n{}'.format(best_clf.__class__.__name__));
plt.ylabel('True')
plt.xlabel('Predicted')
```

    array([[0.94274476, 0.05725524],
           [0.34892251, 0.65107749]])





    Text(0.5, 12.5, 'Predicted')

![png](output_44_2.png)

### Question 5 - Final Model Evaluation

- What is your optimized model's accuracy and F-score on the testing data?
- Are these scores better or worse than the unoptimized model?
- How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?\_

**Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.

#### Results:

|     Metric     | Unoptimized Model | Optimized Model |
| :------------: | :---------------: | :-------------: |
| Accuracy Score |      0.8648       |     0.8724      |
|    F-score     |      0.7386       |     0.7527      |

**Answer:**

- Accuracy of optimized model is 0.8724 and F-Score is 0.7527
- Thse score are nearly same as the un-optimized score. Just a little bit better.
- These scores are way above the score of the naive predictor in Question 1. There the Accuracy score: 0.2478, F-score: 0.2917.

---

## Feature Importance

An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.

Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier. In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.

### Question 6 - Feature Relevance Observation

When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?

```python
# Just to reiterate the features
display(data.columns)
```

    Index(['age', 'workclass', 'education_level', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'],
          dtype='object')

**Answer:**

Following are the 5 features I believe are important from personal experience:

- `education_level`: Gets the level of education. This is important for the earning.
- `occupation` : The occupation also matter. Some occupations have better pay and some do not. Regardless of education this can be an variable that influences.
- `capital-gain`: Regardless of the above 2 the captail gain will determine if the person can pay for charity. If this is not there we cant expect much.
- `captail-loss`: Same as above. This also influence the capability to pay.
- `age`: I guess age plays a role to tell in what kind of position the person is financially.

### Implementation - Extracting Feature Importance

Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.

In the code cell below, you will need to implement the following:

- Import a supervised learning model from sklearn if it is different from the three used earlier.
- Train the supervised model on the entire training set.
- Extract the feature importances using `'.feature_importances_'`.

```python
# TODO: Import a supervised learning model that has 'feature_importances_'

# Fortunately the GraientBoosting has feature_importances_
# TODO: Train the supervised model on the training set using .fit(X_train, y_train)
model = best_clf

# TODO: Extract the feature importances using .feature_importances_
importances = best_clf.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)
```

![png](output_53_0.png)

### Question 7 - Extracting Feature Importance

Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.

- How do these five features compare to the five features you discussed in **Question 6**?
- If you were close to the same answer, how does this visualization confirm your thoughts?
- If you were not close, why do you think these features are more relevant?

**Answer:**

I was not expecting the `martial-status` to impact the individuals ability to earn. But from data it looks like it makes difference. The rest of the features are as I expected.

### Feature Selection

How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to _reduce the feature space_ and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set _with only the top five important features_.

```python
# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
start = time()
clf = (clone(best_clf)).fit(X_train_reduced, y_train)
end = time()
print("Time taken for training: {}".format(end-start))
# Make new predictions
start = time()
reduced_predictions = clf.predict(X_test_reduced)
end = time()
print("Time taken for predicting: {}".format(end-start))

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
print("Time taken for ")
```

    Time taken for training: 1.6157686710357666
    Time taken for predicting: 0.01609182357788086
    Final Model trained on full data
    ------
    Accuracy on testing data: 0.8724
    F-score on testing data: 0.7527

    Final Model trained on reduced data
    ------
    Accuracy on testing data: 0.8627
    F-score on testing data: 0.7347
    Time taken for

### Question 8 - Effects of Feature Selection

- How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?
- If training time was a factor, would you consider using the reduced data as your training set?

**Answer:**

|     Metric     | Optimized Model | Feature Reduced Model |
| :------------: | :-------------: | :-------------------: |
| Accuracy Score |     0.8724      |        0.8627         |
|    F-score     |     0.7527      |        0.7347         |

- Checking the table we can see that the accuracy is not much changing with reduced feature model.
- The time taken for prediction changed from 0.03 to 0.015. This will be very useful when we have a lot more data and more feature to choose from.
- From doing the gridsearch and prediction in this exercise which almost melted my laptop, I believe only keeping the most essential features makes more sense when we are computing in constrained environment.

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
> **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

**Reference**

- [Log Transformations for Skewed and Wide Distributions](https://www.r-statistics.com/2013/05/log-transformations-for-skewed-and-wide-distributions-from-practical-data-science-with-r/)
- [Data Transformations](<https://en.wikipedia.org/wiki/Data_transformation_(statistics)>)
- [Few types of Regression](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/)
- [Marketing Analytics example](http://ucanalytics.com/blogs/marketing-analytics-retail-case-study-part-1/)
- [Decision-trees vs clustering-algorithms vs linear regression](https://dzone.com/articles/decision-trees-vs-clustering-algorithms-vs-linear)
- [Introduction to logistic Regression](https://www.analyticsinsight.net/introduction-to-logistic-regression/)
- [Cancer detection using logistic regression](https://ayearofai.com/rohan-1-when-would-i-even-use-a-quadratic-equation-in-the-real-world-13f379edab3b)
- [Machine Learning in Action](https://www.amazon.in/Machine-Learning-Action-Peter-Harrington/dp/1617290181)
- [Investigation of risk factors associated with injuries to horses undertaking jump racing in Great Britain](http://theses.gla.ac.uk/4624/56/2013reardonphd.pdf)
- [Prognostic Value and Development of a Scoring System in Horses With Systemic Inflammatory Response Syndrome](https://pdfs.semanticscholar.org/150b/255297479d2d486abebf5cd235526e00357e.pdf)
- [The Pros and Cons of Logistic Regression Versus Decision Trees in Predictive Modeling](https://www.huffingtonpost.com/entry/the-pros-and-cons-of-logistic-regression-versus-decision_us_594330ffe4b0d188d027fd1d)
- [Logistic regression vs Descision Tree](https://www.edvancer.in/logistic-regression-vs-decision-trees-vs-svm-part1/)
- [Logistic regression 101](https://machinelearning-blog.com/2018/04/23/logistic-regression-101/)
- [Why use odds ratio in logistic regression](https://www.theanalysisfactor.com/why-use-odds-ratios/)
- [Some problems with using logistic regression](https://www.quora.com/What-are-some-problems-with-the-use-of-logistic-regression)
- [What algorithms are used at work for Big Data](https://www.quora.com/Big-Data/What-algorithms-do-data-scientists-actually-use-at-work-What-are-good-resources-to-learn-the-same/answer/Vijay-Krishnan-1?share=1&srid=37n5)
- [Implement logistic regression with Stochastic gradient descent](https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/)
- [SVM Tutorial](https://www.cse.iitk.ac.in/users/purushot/slides/svm-iiita09.pdf)
- [Application of SVM](https://data-flair.training/blogs/applications-of-svm/)
- [Application of support vector machine for classification of multispectral data](http://iopscience.iop.org/article/10.1088/1755-1315/20/1/012038/pdf)
- [SVM Tutorial](https://www.kdnuggets.com/2017/08/support-vector-machines-learning-svms-examples.html)
- [SVM in environemtal science](https://www.sciencedirect.com/science/article/pii/S0098300412002269)
- [SVM in environmental science](https://www.slideshare.net/beniamino/kernel-based-models-for-geo-and-environmental-sciences-alexei-pozdnoukhov-national-centre-for-geocomputation-national-university-of-ireland-maynooth-ireland)
- [Daily river flow forecasting with SVM](https://pure.uniten.edu.my/en/publications/daily-river-flow-forecasting-with-hybrid-support-vector-machine-p)
- [Kernel methods for geoscience](https://www.slideshare.net/beniamino/kernel-based-models-for-geo-and-environmental-sciences-alexei-pozdnoukhov-national-centre-for-geocomputation-national-university-of-ireland-maynooth-ireland)
- [Classification of Study Region in Environmental Science](http://aclweb.org/anthology/U12-1016)
- [Application of Support Vector Machines for Landuse Classification Using High-Resolution RapidEye Images](https://www.tandfonline.com/doi/pdf/10.5721/EuJRS20154823)
- [Support Vector Machines (SVM) as a Technique for Solvency Analysis](https://core.ac.uk/download/pdf/6302770.pdf)
- [Introduction to support vector machines](https://profs.info.uaic.ro/~ciortuz/SLIDES/svm.pdf)
- [Kaggle Gradient boosting tutorial](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
- [Gradient boosting machines a tutorial](http://www.readcube.com/articles/10.3389/fnbot.2013.00021)
- [Gradient boosting vs random forest](https://medium.com/@aravanshad/gradient-boosting-versus-random-forest-cfa3fa8f0d80)
- [Understanding gradient boosting machines](https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab)
- [Predict survival for cancer patient: simple model or advanced machine learning?](https://www.psiweb.org/docs/default-source/default-document-library/guiyuan-lei-slides.pdf?sfvrsn=2526dedb_0)
- [XGBoost: A Scalable Tree Boosting System](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
- [Pros and Cons of top predective algorithms](https://thereputationalgorithm.com/2017/01/21/top-predictive-algorithms-infographic/)
- [A comparison of Gradient Boosting with Logistic Regression in Practical Cases](https://www.sas.com/content/dam/SAS/support/en/sas-global-forum-proceedings/2018/1857-2018.pdf)
- [Tree Models and Ensembles: Decision Trees, Boosting, Bagging, Gradient Boosting](https://www.youtube.com/watch?v=PGITM1E2CLk)
- [Gradient Boosting from scratch](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)
- [What is the difference between bagging and boosting](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)
- [theory or errors](https://www.encyclopediaofmath.org/index.php/Errors,_theory_of)

```python

```
