import matplotlib.pyplot as plt
import seaborn as sns

def advanced_eda(df):

    print("\n--- DATA SUMMARY ---")
    print(df.describe())

    # Correlation Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # Traffic vs Hour
    if 'Hour' in df.columns:
        plt.figure()
        df.groupby('Hour')['traffic'].mean().plot()
        plt.title("Traffic vs Hour")
        plt.show()

    # Traffic vs DayOfWeek
    if 'DayOfWeek' in df.columns:
        plt.figure()
        df.groupby('DayOfWeek')['traffic'].mean().plot()
        plt.title("Traffic vs DayOfWeek")
        plt.show()

    # Distribution
    plt.figure()
    sns.histplot(df['traffic'], kde=True)
    plt.title("Traffic Distribution")
    plt.show()

    # Traffic vs Weather
    if 'weather' in df.columns:
        plt.figure()
        df.groupby('weather')['traffic'].mean().plot(kind='bar')
        plt.title("Traffic vs Weather")
        plt.show()

    # Traffic vs Holiday
    if 'is_holiday' in df.columns:
        plt.figure()
        df.groupby('is_holiday')['traffic'].mean().plot(kind='bar')
        plt.title("Traffic vs Holiday")
        plt.show()