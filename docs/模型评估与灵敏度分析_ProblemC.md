以下为四问的系统评估与灵敏度分析（基于已生成结果），正文与代码分开呈现。

---

## 问题一：观众投票估计模型
**有效性**：在可处理周次中，规则一致性为 1.0（见 [outputs/eval_q1_summary.csv](../outputs/eval_q1_summary.csv)），表明估计投票与淘汰规则完全一致，满足问题目标。  
**合理性**：投票份额均值为 0.1177、标准差 0.0694、最小值 0、最大值 0.5，区间与参赛人数变化相符。跨周平滑偏好后，赛季内周际平均绝对变化均值为 0.0443，中位数 0.0406（见 [outputs/eval_q1_by_season.csv](../outputs/eval_q1_by_season.csv)），体现序列稳定性。  
**灵敏度**：按裁判评分离散度划分的低/高波动周次，规则一致性均为 1.0（见 [outputs/eval_q1_sensitivity.csv](../outputs/eval_q1_sensitivity.csv)），说明估计对输入波动具有一致的规则复现能力。  
**区间不确定性**：基于“平滑性与相关性偏好权重”的敏感性计算投票区间，得到区间宽度的均值与分位数均为 0（见 [outputs/eval_q1_interval_summary.csv](../outputs/eval_q1_interval_summary.csv)），说明在给定权重扰动范围内估计结果对偏好权重不敏感，区间收敛为点估计。  
**一致性扩展指标**：新增“淘汰预测准确率”与“综合排名偏差”输出（见 [outputs/fan_vote_consistency_q1a.csv](../outputs/fan_vote_consistency_q1a.csv)），在多淘汰周次中以“末 k 一致性”进行检验，用于补充规则一致性检验的可解释性。  
**不确定性扩展指标**：排名法输出可行解数量与排名波动（见 [outputs/fan_vote_uncertainty_rank_q1a.csv](../outputs/fan_vote_uncertainty_rank_q1a.csv)），百分比法输出可行区间与邻域稳定性（见 [outputs/fan_vote_uncertainty_percent_q1a.csv](../outputs/fan_vote_uncertainty_percent_q1a.csv)），并用热力图展示赛季—周次层面的不确定性分布（见 [figures/fan_vote_uncertainty_heatmap_q1a.png](../figures/fan_vote_uncertainty_heatmap_q1a.png)）。  
**稳定性与鲁棒性**：周际变化在赛季之间存在差异（最大值 0.1226），但整体处于低—中幅度范围，说明估计序列稳定且不会出现无依据的剧烈跳动。  
**适用范围与局限**：无人淘汰或多人淘汰周次仍需跳过；最小值为 0 反映部分周次存在极低份额估计，表明在约束强或样本规模小的周次估计可能更集中。

---

## 问题二：规则比较与争议反事实
**有效性**：周次层面共比较 221 周，差异周次 59 个，差异比例 0.26697（见 [outputs/eval_q2_summary.csv](../outputs/eval_q2_summary.csv)），说明模型能实现规则对照目标并量化分歧。  
**合理性**：差异比例非均匀分布，符合“规则作用随赛季结构变化而异”的背景判断。  
**灵敏度**：在低/高裁判评分离散度分组中，差异比例分别为 0.2613 与 0.2727（见 [outputs/eval_q2_sensitivity.csv](../outputs/eval_q2_sensitivity.csv)），差异幅度有限，说明规则分歧对评分波动的敏感性中等。  
**稳定性与鲁棒性**：总体差异比例稳定，但具有赛季异质性；争议个案的反事实路径已生成，可支持情境化分析。  
**适用范围与局限**：结论基于“可比较周次”，对特殊淘汰结构（无人/多人淘汰）仍需单独处理。

---

## 问题三：舞者与选手特征影响
**有效性**：基准拟合度为 $R^2_{judge}=0.3783,\ R^2_{fan}=0.2035$，说明结构变量对裁判评分解释度高于投票份额。  
**合理性**：舞者固定效应与选手特征同时进入，可分离技术与人气影响，符合背景中“裁判评分更技术、观众投票更主观”的逻辑。  
**灵敏度**：留一季重估显示 $R^2_{judge}$ 均值 0.3813、$R^2_{fan}$ 均值 0.2059，标准差分别为 0.0071 与 0.0069（见 [outputs/eval_q3_leave_one_season.csv](../outputs/eval_q3_leave_one_season.csv)），表明模型对样本扰动稳定。  
**稳定性与鲁棒性**：舞者系数的跨季标准差均值为 0.7057（裁判）与 0.00586（投票）（见 [outputs/eval_q3_coef_stability.csv](../outputs/eval_q3_coef_stability.csv)），显示裁判得分效应更敏感、投票效应更平稳，符合两类结果的性质差异。  
**适用范围与局限**：模型可稳定描述平均效应，但对极端人气型选手的个体偏离仍需在解释时保留边界。

---

## 问题四：公平合成系统
**有效性**：公平规则在周次层面可生成唯一淘汰结果，并与输出文件 [outputs/fair_rule_weekly_q4.csv](../outputs/fair_rule_weekly_q4.csv) 一致。  
**合理性**：采用占比规范化与线性加权，确保裁判与观众在同一量纲上共同决策，满足公平性与可解释性要求。  
**灵敏度**：相对于 $\alpha=0.5$ 基准，$\alpha=0.3$ 与 $\alpha=0.7$ 的淘汰差异比例分别为 0.5113 与 0.4570（见 [outputs/eval_q4_sensitivity.csv](../outputs/eval_q4_sensitivity.csv) 与 [figures/eval_q4_alpha_sensitivity.png](../figures/eval_q4_alpha_sensitivity.png)），表明权重调整会显著改变淘汰路径。  
**稳定性与鲁棒性**：在等权结构下规则稳定；但当权重偏离 0.5 时，结果对参数敏感，说明公平规则的可解释性依赖权重设定的透明性与稳定性。  
**适用范围与局限**：适用于强调平衡技术与人气的场景；若节目定位偏向单一目标，应在权重上作明确政策选择。

---

## 代码与输出说明
评估与灵敏度分析代码： [scripts/model_evaluation_sensitivity.py](../scripts/model_evaluation_sensitivity.py)

主要输出文件：
- [outputs/eval_q1_summary.csv](../outputs/eval_q1_summary.csv)
- [outputs/eval_q1_by_season.csv](../outputs/eval_q1_by_season.csv)
- [outputs/eval_q1_sensitivity.csv](../outputs/eval_q1_sensitivity.csv)
- [outputs/eval_q1_interval.csv](../outputs/eval_q1_interval.csv)
- [outputs/eval_q1_interval_summary.csv](../outputs/eval_q1_interval_summary.csv)
- [outputs/fan_vote_consistency_q1a.csv](../outputs/fan_vote_consistency_q1a.csv)
- [outputs/fan_vote_uncertainty_rank_q1a.csv](../outputs/fan_vote_uncertainty_rank_q1a.csv)
- [outputs/fan_vote_uncertainty_percent_q1a.csv](../outputs/fan_vote_uncertainty_percent_q1a.csv)
- [figures/fan_vote_uncertainty_heatmap_q1a.png](../figures/fan_vote_uncertainty_heatmap_q1a.png)
- [outputs/eval_q2_summary.csv](../outputs/eval_q2_summary.csv)
- [outputs/eval_q2_sensitivity.csv](../outputs/eval_q2_sensitivity.csv)
- [outputs/eval_q3_leave_one_season.csv](../outputs/eval_q3_leave_one_season.csv)
- [outputs/eval_q3_coef_stability.csv](../outputs/eval_q3_coef_stability.csv)
- [outputs/eval_q4_sensitivity.csv](../outputs/eval_q4_sensitivity.csv)
- [figures/eval_q4_alpha_sensitivity.png](../figures/eval_q4_alpha_sensitivity.png)
