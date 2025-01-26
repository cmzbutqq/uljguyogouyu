# UNTESTED CODE FROM DEEPSEEK
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# 1. 生成模拟数据（假设历史数据）
def generate_simulated_data(n_countries=200, n_olympics=5):
    """
    生成模拟数据，包含以下特征：
    - country_id: 国家ID
    - gdp_per_capita: 人均GDP（对数正态分布）
    - population: 人口基数（对数正态分布）
    - athletes_count: 参赛运动员数量（与人口正相关）
    - prev_participations: 往届参赛次数
    - is_host: 是否为主办国（二分类）
    - has_won_medal: 是否首次获奖（标签）
    """
    np.random.seed(42)
    data = []
    for olympic in range(n_olympics):
        for country in range(n_countries):
            gdp = np.exp(np.random.normal(8, 1))  # 模拟GDP
            population = np.exp(np.random.normal(14, 0.5))  # 模拟人口
            athletes = int(max(1, np.random.poisson(population * 0.0001)))  # 运动员数量
            participations = np.random.randint(1, 10)  # 往届参赛次数
            is_host = 1 if np.random.rand() < 0.05 else 0  # 5%概率为主办国
            # 生成标签：首次获奖概率与GDP、运动员数量、主办国相关
            prob = 0.01 * (gdp / 1e4) + 0.02 * athletes + 0.3 * is_host
            has_won = 1 if np.random.rand() < prob else 0
            data.append(
                [country, gdp, population, athletes, participations, is_host, has_won]
            )
    df = pd.DataFrame(
        data,
        columns=[
            "country_id",
            "gdp",
            "population",
            "athletes",
            "participations",
            "is_host",
            "has_won",
        ],
    )
    return df


# 生成模拟历史数据（5届奥运会，200个国家）
historical_data = generate_simulated_data()
print("模拟数据示例：\n", historical_data.head())

# 2. 数据预处理
# 移除已获奖国家（假设一旦获奖，后续不再作为正样本）
historical_data = historical_data.groupby("country_id").filter(
    lambda x: x["has_won"].cumsum().eq(0).all()
)

# 特征与标签
X = historical_data[["gdp", "population", "athletes", "participations", "is_host"]]
y = historical_data["has_won"]

# 3. 处理类别不平衡（调整类别权重）
model = make_pipeline(
    StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 模型评估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("\n模型评估：")
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.2f}")
print("混淆矩阵：\n", confusion_matrix(y_test, y_pred))


# 6. 预测下一届奥运会
def predict_next_olympics(model, n_candidate_countries=50):
    # 生成候选国家特征（从未获奖的国家）
    candidate_data = generate_simulated_data(
        n_countries=n_candidate_countries, n_olympics=1
    )
    X_candidate = candidate_data[
        ["gdp", "population", "athletes", "participations", "is_host"]
    ]
    # 计算每个国家的获奖概率
    candidate_data["win_prob"] = model.predict_proba(X_candidate)[:, 1]
    # 计算预期首次获奖国家数
    expected_wins = candidate_data["win_prob"].sum()
    return expected_wins


# 预测结果
expected_wins = predict_next_olympics(model)
print(f"\n预测结果：下一届奥运会预计有 {expected_wins:.1f} 个国家首次获奖。")

# 置信区间（假设泊松分布）
alpha = 0.95
lower = np.percentile(np.random.poisson(expected_wins, 10000), (1 - alpha) * 100)
upper = np.percentile(np.random.poisson(expected_wins, 10000), alpha * 100)
print(f"95% 置信区间：[{lower:.1f}, {upper:.1f}]")
