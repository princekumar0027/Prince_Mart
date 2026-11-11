
# ==========================================
# PROJECT: AI CUSTOMER SUPPORT COPILOT
# TASK 1: DATA LOADING AND EXPLORATION
# ==========================================

# 1. Import Pandas
import pandas as pd

# 2. Load the dataset
df = pd.read_csv(
    r"E:\ML Project 1\data\customer_support_tickets.csv"
)

# 3. Display first five records
print("\n--- FIRST FIVE RECORDS ---")
print(df.head())

# 4. Display last five records
print("\n--- LAST FIVE RECORDS ---")
print(df.tail())

# 5. Check dataset dimensions
print("\n--- DATASET DIMENSIONS ---")
print(df.shape)

# 6. Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 7. Check data types and non-null values
print("\n--- DATASET INFORMATION ---")
df.info()

# 8. Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 9. Check duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())



# ==========================================
# TASK 2: CUSTOMER SUPPORT DATA ANALYSIS
# ==========================================

# 1. Ticket Type Distribution
print("\n--- TICKET TYPE DISTRIBUTION ---")
print(df["Ticket Type"].value_counts())

# 2. Ticket Priority Distribution
print("\n--- TICKET PRIORITY DISTRIBUTION ---")
print(df["Ticket Priority"].value_counts())

# 3. Ticket Status Distribution
print("\n--- TICKET STATUS DISTRIBUTION ---")
print(df["Ticket Status"].value_counts())

# 4. Ticket Channel Distribution
print("\n--- TICKET CHANNEL DISTRIBUTION ---")
print(df["Ticket Channel"].value_counts())

# 5. Product-wise Complaints
print("\n--- PRODUCT-WISE COMPLAINTS ---")
print(df["Product Purchased"].value_counts())

# 6. Customer Satisfaction Rating
print("\n--- CUSTOMER SATISFACTION ---")
print(df["Customer Satisfaction Rating"].value_counts().sort_index())

# 7. Top 10 Ticket Subjects
print("\n--- TOP 10 TICKET SUBJECTS ---")
print(df["Ticket Subject"].value_counts().head(10))

# ==========================================
# END OF TASK 2
# ==========================================


# ==========================================
# TASK 3: MISSING VALUE ANALYSIS
# ==========================================

# Check missing values before treatment
print("\n--- MISSING VALUES BEFORE TREATMENT ---")
print(df.isnull().sum())

# Create a copy of the dataset
df_clean = df.copy()

# Fill missing text fields with a meaningful label
df_clean["Resolution"] = df_clean["Resolution"].fillna(
    "Not Available"
)

df_clean["First Response Time"] = df_clean["First Response Time"].fillna(
    "Not Recorded"
)

df_clean["Time to Resolution"] = df_clean["Time to Resolution"].fillna(
    "Not Recorded"
)

# Keep missing satisfaction ratings as NaN
# because an unknown rating should not be invented.

# Check missing values after treatment
print("\n--- MISSING VALUES AFTER TREATMENT ---")
print(df_clean.isnull().sum())

# Check dataset dimensions
print("\n--- CLEANED DATASET DIMENSIONS ---")
print(df_clean.shape)

# Save cleaned dataset
df_clean.to_csv(
    r"E:\ML Project 1\data\customer_support_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")


# ==========================================
# TASK 4: DATA VISUALISATION
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 1. Ticket Type Distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df_clean, x="Ticket Type",
              order=df_clean["Ticket Type"].value_counts().index)
plt.title("Distribution of Customer Ticket Types")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Ticket Priority Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x="Ticket Priority",
              order=df_clean["Ticket Priority"].value_counts().index)
plt.title("Distribution of Ticket Priorities")
plt.tight_layout()
plt.show()

# 3. Ticket Status Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x="Ticket Status",
              order=df_clean["Ticket Status"].value_counts().index)
plt.title("Distribution of Ticket Status")
plt.tight_layout()
plt.show()

# 4. Customer Satisfaction Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x="Customer Satisfaction Rating")
plt.title("Customer Satisfaction Ratings")
plt.tight_layout()
plt.show()

# ==========================================
# END OF TASK 4
# ==========================================
