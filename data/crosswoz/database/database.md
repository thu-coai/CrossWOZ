# database

- 值缺失一律为 None，导出到json中显示为null，从json导入后是None。
- 周边xx具有对称性，A在B的周边里则B也在A的周边里。条目较多，显示时可截取前五个。
- 门票、评分、人均消费、价格 要用区间查询，支持 小于"<x"|大于">x"|区间(包含端点)"x-y"。
- *: 允许查询的内容。string 类型用字符串匹配，list of string 逐个匹配，int/float 涉及大小比较。**推荐菜**和**酒店设施**支持多个条件匹配，用空格分隔，如 "东北杀猪菜 锅包肉"，检索时要求两个都出现在推荐菜中。
- 出租数据库是模板，不查询，均为占位符。

### 景点

- 领域: "景点"
- 名称*: string
- 地址: string
- 地铁: string
- 电话: string
- 门票*: int (缺失则为None)
- 游玩时间*: string
- 评分*: float (缺失则为None)
- 周边景点*: list of string
- 周边餐馆*: list of string
- 周边酒店*: list of string



### 餐馆

- 领域: "餐馆"
- 名称*: string
- 地址: string
- 地铁: string
- 电话: string
- 营业时间: string
- 推荐菜*: list of string
- 人均消费*: int
- 评分*: float (缺失则为None)
- 周边景点*: list of string
- 周边餐馆*: list of string
- 周边酒店*: list of string



### 酒店

- 名称*: string
- 酒店类型*: string
- 地址: string
- 地铁: string
- 电话: string
- 酒店设施*: list of string
- 价格*: int (缺失则为None)
- 评分*: float
- 周边景点*: list of string
- 周边餐馆*: list of string
- 周边酒店*: list of string




### 地铁

- 名称*: string
- 地铁*: string (缺失则为None)



### 出租

- 车型: "#CX"
- 车牌: "#CP"
