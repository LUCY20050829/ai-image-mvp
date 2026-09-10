# 人物脸图说明

## 目录用途

- `lower_primary`：小低阶段脸图，来自《小低脸图.xlsx》中可提取的 WPS `DISPIMG` 嵌入图。
- `middle_upper_primary`：原有脸图库，目前按小中小高阶段使用并重命名复制。
- 根目录下的旧脸图：暂时保留作为回落查找，不直接删除。

## 后端查找规则

`server.py` 会根据 `grade_band` 优先查找对应学段目录：

- `lower_primary`：先查 `lower_primary`，再回落旧根目录。
- `middle_upper_primary`：先查 `middle_upper_primary`，再回落旧根目录。
- `junior_high`：目前暂无独立脸图库，先查 `junior_high`，再回落 `middle_upper_primary` 和旧根目录。

## 小低缺图

《小低脸图.xlsx》中以下人物未提取到嵌入图，需后续补充：孟浩然、贾岛、白居易、杨万里。
