# A Data Labelling System for CoAI Lab, May 2019

## Requirements

* python3.6

## Development Startup

Copy `.env.example` to `.env`, `settings.example.py` to `settings.py` and update them with local config.

Run the following command and visit http://localhost:5000.

```bash
# install dependencies
pip install -r requirements.txt

# initialize the database
python resetdb.py

# start the server
python run.py
```

## Deployment Instruction

You can use exactly same steps as development startup.

## 操作指南

启动服务器以后，可以使用默认管理员账户登录。

用户名：root

密码：root

若要进入普通用户界面，请点击注册链接，使用邀请码注册。

当前邀请码为 `959592`，可以修改 `data_labelling/app.py` 的 `invitation_code`。

普通用户登录后，进入对话匹配界面，匹配需要**至少有一人选择系统端，一人选择用户端**，此时系统会自动完成匹配，两人进入对话界面。

提示：如果在本地测试功能，可以使用 Chrome 的无痕窗口同时登录两个用户。

在对话界面，用户端先开始对话，以一问一答的方式进行。某一方发送消息后，另一方可以立即看到，但是只有当前者完成必要的信息标注后，后者才可以发送消息。

对话页面相关逻辑参见 `templates` 目录下的文件。

管理员可以在后台导入预先定义的任务，导出对话数据。

导入任务的步骤是：在 Result Files 选项卡下，进入 inputs 目录，上传任务定义文件 tasks.json，再回到管理首页，点导入按钮。导入成功后，系统会告知导入成功的任务数量。此时也可以到 Task 选项卡查看详情。

导出数据直接在管理首页点下载全部即可。


