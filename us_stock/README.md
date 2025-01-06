# My thoughts

可以搞一个 TMF TQQQ 的 的 monthly or weekly adjusted portfolio

## 数据来源

### USSTOCKEODPRICES: US stock EOD prices

```python
"wind": {
    "host": "10.181.113.239",
    "port": 1433,
    "database": "WindDB",
    "username": "gdsyxypj",
    "password": "g1dsyxypj@Gtja",
}
```

```sql
SELECT OBJECT_ID, S_INFO_WINDCODE, TRADE_DT, CRNCY_CODE, S_DQ_PRECLOSE, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_ADJFACTOR, S_DQ_PRECLOSE_WIND, OPDATE, OPMODE FROM WindDB.dbo.USSTOCKEODPRICES WHERE OBJECT_ID=N'{2E1C8DBD-EB60-40B4-9FC9-0DABD4403A9A}';
```

这个表有所有国际交易所的交易时段




# 美股月频/周频策略动竞赛

## 需求发布日期

2024 年 10 月 16 日

## 需求名称

美股月频/周频策略

## 需求概述

针对美股 ETF 标的（挂钩道琼斯指数、标普 500、纳指 100 或纳斯达克指数等。对于上市时间较短的 ETF 可在 ETF 未上市时间段使用指数本身回测，上市后再用 ETF 替换），基于价量因子或全球宏观数据与美国经济基本面数据等因子，搭建可实操的周频或月频策略模型【仓位为多头或无仓位】。要求交易调仓为可实现方式（如结算价、某段交易时间的平均价等）。需考虑滑点。基本面数据需要保证可定期自动更新，且确保使用数据的时间在发布时间以后，避免使用未来数据。需注意 T 日策略信号出现后，T+1 日完成调仓，T+2 日计算的收益始为调仓后收益。策略需要区分训练集与测试集。

### 需求发布方

QIS 业务团队

### 考核支付标准

50 万

### 策略收录标准/评价标准

1. 测试集长度至少 5 年，且至少每年需滚动重训练模型；

2. 要求日频计算近五年夏普率不低于 1.5，控制最大回撤 10%以内，在美股下跌的年份策略收益必须高于美股指数本身。提交后样本外半年考核的夏普率大于等于 1.2。

### 响应方案提交截止日期

需求发布后 **_4_** 个月内

### 发布方联系人

王康

### 信用中心联系人

宋悉瑜
