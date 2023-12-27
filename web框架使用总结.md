# 前言

算法工程师在将AI模型部署上线时，通常需要将模型的推理功能封装成web服务，供前后端人员或者用户来调用模型训练、模型推理、模型评测等功能，这篇文章介绍了python常用的http web框架、rpc web框架和日志脚本

# 1.flask

## 1.安装

``` bash
pip install flask
```

## 2.service

```python
from flask import Flask, request
import json

# 实例化flask app
app = Flask(__name__)

# 定义路由和允许的请求方法，可以允许多个请求方法
@app.route("/index", methods=["POST", "GET"])
def get_info():
    # 获取请求体参数，并将json格式反序列化成python字典或列表
    request_data = json.loads(request.get_data())
    print("type: ", type(request_data), "\ndata: ", request_data)

    return_dict = {
        "code": 0,
        "msg": "good",
        "data": request_data
    }  
    return json.dumps(return_dict) # 将响应结果序列化成json再返回

if __name__ == '__main__':
  	# 设置服务的host地址和端口号
    # app.run(host="0.0.0.0", port=5000)  # ipv4
    app.run(host="::", port=5000)  # ipv6
```

## 3.query

```python
import requests
import json

# url = "http://127.0.0.1:5000/index" # ipv4
url = "http://[::1]:5000/index"  # ipv6
payload = json.dumps({
    "C": 3,
    "D": 4
})

headers = {'Content-Type': 'application/json'}

# 将请求体参数payload序列化成json格式后通过request的data参数传入
response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```

## 4.return

```bash
{"code": 0, "msg": "good", "data": {"C": 3, "D": 4}}
```

# 2.fastapi

## 1.安装

```bash
pip install fastapi
pip install uvicorn
```

## 2.service

```python
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Literal, Optional, List
import json

# 实例化fastapi app
app = FastAPI()

# 定义请求参数
class requestParams(BaseModel):
    A: int  # int类型的参数
    B: Optional[str] = "aaa"  # 可选参数，string类型，需设置默认值
    C: List[int | str]  # list类型的参数，list里面可以是int或者str
    D: Literal["a", 'b', 'c']  # 枚举类型的参数，只能传着三个值之一


# 定义路由和允许的请求方法
@app.post("/index")
def get_info(request_params: requestParams):  # 规定请求参数的类型，fastapi会自动校验参数
    # 将请求体参数转成python字典
    request_params_dict = request_params.model_dump()
    print("request_params_dict: ", request_params_dict)
    # request_params是一个对象，可以直接访问对象属性
    print("A: ", request_params.A)
    return_dict = {
        "code": 0,
        "msg": "good",
        "data": request_params_dict
    }
    return json.dumps(return_dict)


if __name__ == '__main__':
    # uvicorn.run(app=app, host='0.0.0.0', port=8000) # ipv4
    uvicorn.run(app=app, host='::', port=8000)  # ipv6
```

## 3.query

```python
import requests
import json

# url = "http://127.0.0.1:8000/index"# ipv4
url = "http://[::1]:8000/index"  # ipv6

payload = json.dumps({
    "A": 3,
    "C": [1, 'a'],
    "D": 'b'
})

headers = {'Content-Type': 'application/json'}

# 将请求体参数payload序列化成json格式后通过request的data参数传入
response = requests.request("POST", url, headers=headers, data=payload)

print(json.loads(response.text))
```

## 4.return

```bash
# 可以看到，虽然请求里没有传参数B，但是返回了参数B的默认值
{"code": 0, "msg": "good", "data": {"A": 3, "B": "aaa", "C": [1, "a"], "D": "b"}}
```

# 3.thriftpy2

Thriftpy2是第三方对thrift的纯python封装，不同手动编译来生成service和client代码，写法更简单一些，

## 1.安装

```bash
pip install thriftpy2
```

## 2.定义IDL（Interface Description Language）文件

```thrift
//基本数据类型
//bool: A boolean value (true or false)
//byte: An 8-bit signed integer
//i16: A 16-bit signed integer
//i32: A 32-bit signed integer
//i64: A 64-bit signed integer
//double: A 64-bit floating point number
//string: A text string encoded using UTF-8 encoding
//list<data type>:数组
//set<data type>:集合
//map<key type, value type>:字典


//initPersonInfo方法的请求参数
struct personInfo{
    1:required string name; //姓名
    2:required i32 age; //年龄
    3:required double height; //身高
    4:required bool is_man; //是否男性
    5:required list<string> hobby; //爱好
    6:required set<double> study_time; //一周内每天的学习时长
    7:required map<string, double> subject_grade; //科目和成绩
}

struct returnFormat{ //成功时的返回值
    1:required i32 code;
    2:required string msg;
    3:required map<string, string> data;
}

exception Xecption{ //失败时的返回值
    1:required i32 code;
    2:required string msg;
}

// 定义服务，其中含有两个方法
service PersonInfoService{
    returnFormat initPersonInfo(1:required personInfo person_info) throws (1:Xecption xecption);
    returnFormat getPersonInfo(1:required string feature) throws (1:Xecption xecption);
}
```

## 3.service

```python
import thriftpy2
from thriftpy2.rpc import make_server

# 导入idl文件
service = thriftpy2.load(path='idl.thrift', module_name="person_info_thrift")
return_format = service.returnFormat
exception_format = service.Xecption


class PersonInfoHandler: # 定义service的handler，这个例子中，service有两个方法，分别是initPersonInfo和getPersonInfonfo
    def initPersonInfo(self, person_info):
        print("person_info:", person_info)
        try:
            self.person_info = person_info

            return_format.code = 0
            return_format.msg = "init info succeed"
            return_format.data = {"result": "init succeed"}
            return return_format
        except Exception as e:
            exception_format.code = 500
            exception_format.msg = str(e)
            return exception_format

    def getPersonInfo(self, feature):
        print("feature:", feature)
        try:
            match feature:
                case "name":
                    res = self.person_info.name
                case "age":
                    res = self.person_info.age
                case "height":
                    res = self.person_info.height
                case "is_man":
                    res = self.person_info.is_man
                case "hobby":
                    res = self.person_info.hobby
                case "study_time":
                    res = self.person_info.study_time
                case "subject_grade":
                    res = self.person_info.subject_grade
                case _: # _ 代表默认情况下的值
                    res = "do not have the feature"

            return_format.code = 0
            return_format.msg = "succeed"
            return_format.data = {"result": str(res)}
            return return_format
        except Exception as e:
            exception_format.code = 500
            exception_format.msg = str(e)
            return exception_format

# 创建service对象，貌似host只支持ipv4，启动服务
server = make_server(service=service.PersonInfoService, handler=PersonInfoHandler(), host='0.0.0.0', port=6000)
print("serving...")
server.serve()

```

## 4.client

```python
import thriftpy2
from thriftpy2.rpc import make_client

# 导入idl文件
service = thriftpy2.load(path='idl.thrift', module_name="person_info_thrift")
# 创建client对象
client = make_client(service=service.PersonInfoService, host='127.0.0.1', port=6000)

# 构造请求参数
person_info = service.personInfo  # 实例化personInfo数据结构
person_info.name = "张三"
person_info.age = 18
person_info.height = 169.5
person_info.is_man = True
person_info.hobby = ["唱", "跳", "打篮球"]
person_info.study_time = [4.5, 2.3, 4.6, 4.6]
person_info.subject_grade = {"语文": 75, "数学": 80, "英语": 90}

# 发送initPersonInfo请求
init_response = client.initPersonInfo(person_info)
print(init_response)

# 发送getPersonInfo请求
feature = "subject_grade"
get_info_response = client.getPersonInfo(feature)
print(get_info_response)

```

## 5.return

```bash
returnFormat(code=0, msg='init info succeed', data={'result': 'init succeed'})
returnFormat(code=0, msg='succeed', data={'result': "{'语文': 75.0, '数学': 80.0, '英语': 90.0}"})
```

# 4.logging

## 1.logging简单配置

这种配置下，一个服务只有一个logging对象，所有就只有一个日志文件

```python
logging.basicConfig(filename="log.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 写日志
logging.inifo("这是一条测试日志")
logging.warning('这是一条warning日志')
```

## 2.logger详细配置

这种配置下，可以在一个服务服务中实例化多个logger对象，可以创建多个日志文件

```python
import logging

formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')  # 日志写入格式：时间-级别-日志内容
handler = logging.FileHandler(filename='./log.log')  # 实例化handler，设置日志文件存放地址
handler.setFormatter(formatter)  # 给handler设置日志格式
handler.setLevel(logging.INFO)  # 给handler设置日志级别，这决定了handler写入文件的最低日志级别
logger = logging.getLogger('main_log')  # 实例化名为main_log的logger
logger.addHandler(handler)  # 给logger添加handler
logger.setLevel(logging.INFO)  # 给logger设置日志级别，这决定了logger发送给handler的最低日志级别

# 写日志
logger.info('这是一条info日志')
logger.warning('这是一条warning日志')
```

# 5.总结

RPC（Remote Procedure Call）远程程序调用，是一种通过网络从远程计算机上请求服务，而不需要了解底层网络技术的协议。常见的RPC框架有[grpc](https://grpc.io/docs/languages/python/quickstart)和[thrift](https://thrift.apache.org)，它们都支持跨语言。开发RPC服务时，首先写好IDL（grpc的文件后缀是.proto ，thrift的文件后缀是.thrift，IDL文件中定义了服务的名称，服务中包含的方法，各个方法的请求参数，服务的返回参数，请求异常时的返回参数等）文件，然后由服务的提供方和调用者根据IDL文件用各自的编程语言分别开发service和client功能