import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# membaca dataset
df = pd.read_csv("dataset_buys_comp.csv")

# menampilkan 5 data pertama
print(df.head())

# mengecek apakah ada data yang hilang
print(df.isnull().sum())

# menampilkan nilai unik dari setiap kolom
print("\nValue Kolom:")
for col in df.columns:
    print(f"{col}: {df[col].unique()}")

# melakukan encoding pada kolom kategorikal
df_encoded = df.copy()
label_encoders = {}

for column in df_encoded.columns[:-1]:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le

# membagi data menjadi fitur dan target
X = df_encoded.drop("Buys_Computer", axis=1)
y = df_encoded["Buys_Computer"]

# membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# membuat dan melatih model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Decision Tree dengan entropy
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)

# evaluasi model entropy
print("\n--- Evaluasi Model Entropy ---")
y_pred_entropy = dt_entropy.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred_entropy))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_entropy))
print("Classification Report:\n", classification_report(y_test, y_pred_entropy))

# visualisasi tree untuk model dengan entropy
plt.figure(figsize=(12, 8))
plot_tree(dt_entropy, feature_names=X.columns, class_names=["Tidak", "Ya"], filled=True)
plt.title("Decision Tree - Model Entropy")
plt.show()

# visualisasi tree untuk model default
plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=X.columns, class_names=["Tidak", "Ya"], filled=True)
plt.title("Decision Tree")
plt.show()

# menampilkan feature importance untuk model default
feature_importance = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# menampilkan feature importance untuk model entropy
feature_importance_entropy = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': dt_entropy.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Entropy):")
print(feature_importance_entropy)

# pengujian dengan data baru
data_baru = {
    'Age': ['Muda'],
    'Income': ['Rendah'],
    'Student': ['Ya'],
    'Credit_Rating': ['Buruk']
}

print("\nData Baru:")
print(data_baru)

# mengubah data baru menjadi DataFrame dan melakukan encoding
df_new = pd.DataFrame(data_baru)
for column in df_new.columns:
    df_new[column] = label_encoders[column].transform(df_new[column])

# prediksi dengan kedua model
hasil_default = dt_model.predict(df_new)
hasil_entropy = dt_entropy.predict(df_new)

print("\nPrediksi data baru:")
print("Model default     :", 'Layak' if hasil_default[0] == 1 else 'Tidak Layak')
print("Model Entropy    :", 'Layak' if hasil_entropy[0] == 1 else 'Tidak Layak')