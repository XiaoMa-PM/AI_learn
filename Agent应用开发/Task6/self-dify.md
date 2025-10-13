# self-dify
---
🧑‍🔧作者：王熠明
🔗link：[self-dify2.0 - 飞书云文档](https://spvrm23ffj.feishu.cn/docx/RLVSdcLZZo7qhjxbsyTcNT2bnf5)


## Dify本地部署
### 安装Docker&换源
前置必须安装docker环境
安装Docker compose

安装后配置Docker Engine

### Dify本地部署docker
目标文件夹打开终端dify-1.6
cd docker
打开文件夹隐藏
command+shift+.
```
    cp .env.example .env 
    docker compose up -d # 第一次会先拉取镜像，所以可能会有点久，可以不加-d，查看日志信息

```

### 升级Dify
下载最新安装包，然后copy原有的dify文件夹下的docker-volumes与.env过去
在docker下打开终端
运行下面两个步骤
```
    cp .env.example .env 
    docker compose up -d # 第一次会先拉取镜像，所以可能会有点久，可以不加-d，查看日志信息

```