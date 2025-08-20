"""
Прогнозирование стоимости автомобилей с помощью Machine Learning
Анализ данных и построение модели предсказания цен
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.float_format', '{:.2f}'.format)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_prepare_data(file_path):
    """Загрузка и подготовка данных"""
    print("🔹 Загрузка данных...")
    df = pd.read_csv(file_path)

    print(f"Размер данных: {df.shape}")
    print(f"Колонки: {list(df.columns)}")

    df = df.drop('car_ID', axis=1)
    return df


def exploratory_data_analysis(df):
    """Разведочный анализ данных"""
    print("\n🔹 Разведочный анализ данных...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(df['price'], bins=40, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Распределение цен на автомобили', fontsize=14)
    axes[0].set_xlabel('Цена ($)', fontsize=12)
    axes[0].set_ylabel('Количество', fontsize=12)
    axes[0].grid(alpha=0.3)

    sns.boxplot(y=df['price'], ax=axes[1], color='lightcoral', showmeans=True)
    axes[1].set_title('Boxplot цен', fontsize=14)
    axes[1].set_ylabel('Цена ($)', fontsize=12)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    stats = df['price'].describe()
    print("\n📊 Статистики цен:")
    print(f"   Мин: ${stats['min']:,.0f}")
    print(f"   Q1: ${stats['25%']:,.0f}")
    print(f"   Медиана: ${stats['50%']:,.0f}")
    print(f"   Q3: ${stats['75%']:,.0f}")
    print(f"   Макс: ${stats['max']:,.0f}")
    print(f"   Среднее: ${stats['mean']:,.0f}")
    print(f"   Стандартное отклонение: ${stats['std']:,.0f}")

    Q1 = stats['25%']
    Q3 = stats['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
    print(f"\n📈 Анализ выбросов:")
    print(f"   Границы выбросов: ${lower_bound:,.0f} - ${upper_bound:,.0f}")
    print(f"   Количество выбросов: {len(outliers)}")
    print(f"   Доля выбросов: {(len(outliers)/len(df))*100:.1f}%")

    skewness = df['price'].skew()
    print(f"\n📊 Скошенность распределения: {skewness:.2f}")

    if skewness > 1:
        print("➡️ Сильно правосторонняя скошенность")
    elif skewness > 0.5:
        print("➡️ Умеренная правосторонняя скошенность")
    elif skewness < -1:
        print("⬅️ Сильно левосторонняя скошенность")
    elif skewness < -0.5:
        print("⬅️ Умеренная левосторонняя скошенность")
    else:
        print("↔️ Близко к нормальному распределению")

    return stats


def correlation_analysis(df):
    """Анализ корреляций"""
    print("\n🔹 Анализ корреляций...")

    numeric_features = [
        'wheelbase', 'carlength', 'carwidth', 'carheight',
        'curbweight', 'enginesize', 'boreratio', 'stroke',
        'compressionratio', 'horsepower', 'peakrpm', 'citympg',
        'highwaympg', 'price'
    ]

    numeric_df = df[numeric_features]
    corr_matrix = numeric_df.corr()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    price_corr = corr_matrix['price'].drop('price').sort_values()
    colors = ['red' if x < 0 else 'green' for x in price_corr.values]

    bars = ax1.barh(price_corr.index, price_corr.values, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.8)

    for i, (bar, value) in enumerate(zip(bars, price_corr.values)):
        ax1.text(value + (0.01 if value >= 0 else -0.05), i, f'{value:.2f}',
                 va='center', fontweight='bold', fontsize=10)

    ax1.set_title('Корреляция с ценой', fontsize=14)
    ax1.set_xlabel('Коэффициент корреляции', fontsize=12)
    ax1.set_ylabel('Признаки', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={'shrink': 0.8},
        ax=ax2
    )
    ax2.set_title('Матрица корреляций', fontsize=14)

    plt.tight_layout()
    plt.show()

    print("\n📊 Анализ корреляций с ценой:")
    print("-" * 40)

    strong_corr = price_corr[(price_corr.abs() > 0.7) & (price_corr.index != 'price')]
    if not strong_corr.empty:
        print("🔹 Сильные корреляции:")
        for feature, corr in strong_corr.items():
            print(f"   {feature}: {corr:.3f}")

    moderate_corr = price_corr[(price_corr.abs() > 0.5) & (price_corr.abs() <= 0.7)]
    if not moderate_corr.empty:
        print("\n🔸 Умеренные корреляции:")
        for feature, corr in moderate_corr.items():
            print(f"   {feature}: {corr:.3f}")

    print("\n💡 Интерпретация:")
    if 'enginesize' in numeric_features and abs(corr_matrix.loc['enginesize', 'price']) > 0.5:
        print("   - Размер двигателя сильно влияет на цену")
    if 'curbweight' in numeric_features and abs(corr_matrix.loc['curbweight', 'price']) > 0.5:
        print("   - Вес автомобиля коррелирует с ценой")

    return numeric_df, corr_matrix


def prepare_model_data(numeric_df):
    """Подготовка данных для модели"""
    print("\n🔹 Подготовка данных для модели...")

    X = numeric_df.drop('price', axis=1)
    y = numeric_df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"Размер X_train: {X_train.shape}")
    print(f"Размер X_test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Обучение и оценка модели"""
    print("\n🔹 Обучение модели...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Метрики качества модели:")
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE: {mae:,.2f}")
    print(f"R² Score: {r2:.4f}")

    mean_price = y_test.mean()
    print(f"\n📈 Относительные ошибки:")
    print(f"RMSE / Средняя цена: {(rmse/mean_price)*100:.1f}%")
    print(f"MAE / Средняя цена: {(mae/mean_price)*100:.1f}%")

    return model, y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def plot_predictions(y_true, y_pred, metrics, model_name='Random Forest'):
    """Визуализация предсказаний модели"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='steelblue', edgecolor='white', linewidth=0.5)

    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Идеальное предсказание')

    plt.xlabel('Реальная цена ($)', fontsize=12)
    plt.ylabel('Предсказанная цена ($)', fontsize=12)
    plt.title(f'{model_name}\nR² = {metrics["r2"]:.3f}', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    errors = y_true - y_pred
    plt.hist(errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='Нет ошибки')
    plt.xlabel('Ошибка предсказания ($)', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.title(f'Распределение ошибок\nMAE = ${metrics["mae"]:,.0f}', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return errors


def analyze_results(y_test, y_pred, errors, feature_names, model):
    """Анализ результатов модели"""
    print("\n🔹 Детальный анализ результатов...")

    results_df = pd.DataFrame({
        'Реальная_цена': y_test.values,
        'Предсказанная_цена': y_pred,
        'Ошибка': errors,
        'Абсолютная_Ошибка': np.abs(errors),
        'Относительная_Ошибка_%': (np.abs(errors) / y_test.values) * 100
    })

    print("\n🎯 Топ-5 самых точных предсказаний:")
    accurate_predictions = results_df.nsmallest(5, 'Абсолютная_Ошибка')
    print(accurate_predictions[['Реальная_цена', 'Предсказанная_цена', 'Абсолютная_Ошибка', 'Относительная_Ошибка_%']].round(2))

    print("\n⚠️ Топ-5 самых больших ошибок:")
    large_errors = results_df.nlargest(5, 'Абсолютная_Ошибка')
    print(large_errors[['Реальная_цена', 'Предсказанная_цена', 'Абсолютная_Ошибка', 'Относительная_Ошибка_%']].round(2))

    print("\n📊 Важность признаков:")
    feature_importance = pd.DataFrame({
        'Признак': feature_names,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=False)

    print(feature_importance.head(10).to_string(index=False))

    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)

    sns.barplot(
        x='Важность',
        y='Признак',
        data=top_features,
        hue='Признак',
        palette='rocket',
        legend=False,
        dodge=False
    )

    plt.title('Топ-10 самых важных признаков', fontsize=16)
    plt.xlabel('Важность признака', fontsize=12)

    for i, (importance, feature) in enumerate(zip(top_features['Важность'], top_features['Признак'])):
        plt.text(importance + 0.001, i, f'{importance:.3f}',
                 va='center', fontweight='bold', fontsize=10)

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n💡 Интерпретация важности признаков:")
    top_3 = feature_importance.head(3)
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"{i}. {row['Признак']}: {row['Важность']:.3f}")

    return results_df, feature_importance


def main():
    """Основная функция"""
    print("🚗 Прогнозирование стоимости автомобилей")
    print("=" * 50)

    try:
        df = load_and_prepare_data('CarPrice_Assignment.csv')
        exploratory_data_analysis(df)
        numeric_df, corr_matrix = correlation_analysis(df)
        X_train, X_test, y_train, y_test, feature_names = prepare_model_data(numeric_df)
        model, y_pred, metrics = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        errors = plot_predictions(y_test, y_pred, metrics)
        results_df, feature_importance = analyze_results(y_test, y_pred, errors, feature_names, model)

        print("\n" + "=" * 50)
        print("✅ Анализ завершен успешно!")
        print(f"🎯 Модель объясняет {metrics['r2']*100:.1f}% дисперсии цен")
        print(f"📏 Средняя ошибка: ${metrics['mae']:,.0f}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()