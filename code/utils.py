import pandas as pd
import numpy as np
from matplotlib.pyplot import colorbar
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Function to load baseball dataset in baseball.py
def load_baseball_data():
    train_raw = pd.read_csv('../data/baseball_train.csv')
    test_raw = pd.read_csv('../data/baseball_test.csv')

    # Return in order X_train, X_scaled_train, y_train, X_test, X_scaled_test, y_test
    return process_baseball(train_raw, test_raw)

# Function to load animal dataset in animal.py
def load_animal_data():
    train_raw = pd.read_csv('../data/animal_train.csv')
    test_raw = pd.read_csv('../data/animal_test.csv')

    # Return in order X_train, X_scaled_train, y_train, X_test, X_scaled_test, y_test
    return process_animal(train_raw, test_raw)

# Function to create the learning curve using scikit-learns learning_curve()
def create_learning_curve(model, X_train, y_train, cv, title, path, scoring='f1'):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring=scoring, random_state = 42, n_jobs=-1)

    # Create arrays of training and validation means from the returned scores, also the stddevs for the shaded regions
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # train_sizes was outputting the number of samples, want percents
    train_sizes = train_sizes / X_train.shape[0]

    # Plot Learning Curve
    plt.figure()
    plt.plot(train_sizes, train_mean, label='Training F1')
    plt.plot(train_sizes, val_mean, label='Validation F1')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Fraction of Training Set')
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()
    return

# Function to create confusion matrices
def plot_confusion_matrix(model, X_test, y_test, title, path, labels):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    disp.plot(cmap='Blues', ax=ax, colorbar=False, text_kw={'fontsize': 12})
    plt.title(title)
    if (len(labels) > 2) :
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

# Function to produce the results table
def plot_results_table(results, columns, title, path):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    plt.title(title)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

# Function to preprocess the baseball dataset
def process_baseball(b_train, b_test):
    # TRANSFORMATIONS

    # Create batter_is_home, 0 or 1
    b_train['batter_is_home'] = (b_train['batter_team'] == b_train['home_team']).astype(int)
    b_test['batter_is_home'] = (b_test['batter_team'] == b_test['home_team']).astype(int)

    # Create same_handedness, 0 or 1
    b_train['same_handedness'] = (b_train['is_batter_lefty'] == b_train['is_pitcher_lefty']).astype(int)
    b_test['same_handedness'] = (b_test['is_batter_lefty'] == b_test['is_pitcher_lefty']).astype(int)

    # Create is_fastball, 0 or 1
    fastball_pitches = ['4-Seam Fastball', 'Sinker', '2-Seam Fastball', 'Cutter']
    b_train['is_fastball'] = (b_train['pitch_name'].isin(fastball_pitches)).astype(int)
    b_test['is_fastball'] = (b_test['pitch_name'].isin(fastball_pitches)).astype(int)

    # Drop missing launch stats
    b_train = b_train.dropna(subset=['launch_speed', 'launch_angle'])
    b_test = b_test.dropna(subset=['launch_speed', 'launch_angle'])

    # One hot encode categorical features of bb_type and bearing
    b_train = pd.get_dummies(b_train, columns=['bb_type', 'bearing'], dtype=int)
    b_test = pd.get_dummies(b_test, columns=['bb_type', 'bearing'], dtype=int)

    # Drop unneeded columns
    cols_to_drop = [
        'bip_id',
        'game_date',
        'batter_name',
        'pitcher_name',
        'batter_id',
        'pitcher_id',
        'park',
        'home_team',
        'away_team',
        'batter_team',
        'pitch_name',
        'is_pitcher_lefty',
        'is_batter_lefty'
    ]

    b_train = b_train.drop(columns=cols_to_drop)
    b_test = b_test.drop(columns=cols_to_drop)

    # Split the training and testing data into features and target
    b_X_train = b_train.drop(columns=['is_home_run'])
    b_y_train = b_train['is_home_run']

    b_X_test = b_test.drop(columns=['is_home_run'])
    b_y_test = b_test['is_home_run']

    # Scale
    scaler = StandardScaler()
    b_X_train_scaled = scaler.fit_transform(b_X_train)
    b_X_test_scaled = scaler.transform(b_X_test)

    # Convert from NumPy array back to Panads DF
    b_X_train_scaled_df = pd.DataFrame(b_X_train_scaled, columns=b_X_train.columns)
    b_X_test_scaled_df = pd.DataFrame(b_X_test_scaled, columns=b_X_test.columns)

    return b_X_train, b_X_train_scaled_df, b_y_train, b_X_test, b_X_test_scaled_df, b_y_test

# Function to convert age strings to days. Used in process_animal()
def age_to_days(age_str):
    if pd.isna(age_str):
      return None
    try:
      num, unit = str(age_str).split()
      num = int(num)
      if 'year' in unit:
        return num * 365
      if 'month' in unit:
        return num * 30
      if 'week' in unit:
        return num * 7
      return num # days
    except:
      return None

# Function to preprocess the animal dataset
def process_animal(a_train, a_test):
    # Transform Ageuponoutcome, Breed, and Name
    for df in [a_train, a_test]:
        df['age_days'] = df['AgeuponOutcome'].apply(age_to_days)
        df['is_purebred'] = df['Breed'].apply(lambda x: 0 if ('Mix' in x or '/' in x) else 1)
        df['has_name'] = df['Name'].notna().astype(int)

    # Drop the 15 train rows with missing ages
    a_train = a_train.dropna(subset=['age_days'])

    # Drop the 4 test rows with missing age_days/SexuponOutcome
    a_test = a_test.dropna(subset=['age_days', 'SexuponOutcome'])

    # One hot encode SexuponOutcome
    a_train = pd.get_dummies(a_train, columns=['SexuponOutcome'], dtype=int)
    a_test = pd.get_dummies(a_test, columns=['SexuponOutcome'], dtype=int)

    # Transform AnimalType to is_dog
    a_train['is_dog'] = (a_train['AnimalType'] == 'Dog').astype(int)
    a_test['is_dog'] = (a_test['AnimalType'] == 'Dog').astype(int)

    # Drop unnecessary rows
    a_train = a_train.drop(
        columns=['Color', 'AnimalType', 'AnimalID', 'DateTime', 'Name', 'AgeuponOutcome', 'OutcomeSubtype', 'Breed'])
    a_test = a_test.drop(
        columns=['Color', 'AnimalType', 'AnimalID', 'DateTime', 'Name', 'AgeuponOutcome', 'OutcomeSubtype', 'Breed'])

    # Separate features and labels
    X_train = a_train.drop(columns=['OutcomeType'])
    X_test = a_test.drop(columns=['OutcomeType'])

    # Encode labels
    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(a_train['OutcomeType']))
    y_test = pd.Series(le.transform(a_test['OutcomeType']))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test