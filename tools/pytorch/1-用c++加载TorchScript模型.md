#### 1.将pytorch模型转换成Torch Script

- 方法一：Tracing

```python
import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
```

- 方法二：Annotation，在forward方法中有控制输入流时，不适合用Tracing

```python
class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

my_module = MyModule(10,20)
sm = torch.jit.script(my_module)
```

#### 2.序列化Torch Script

两种方法生成的Torch Script都可以用save方法序列化模型：

```python
traced_script_module.save("traced_resnet_model.pt")

sm.save("annotated_model.pt")
```

#### 3.用C++加载Torch Script

```c++
#include<torch/script.h>
#include <torch/torch.h>
#include<iostream>
#include<memory>

int main() {
	torch::jit::script::Module module;

	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load("./traced_resnet_model.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	std::cout << "ok\n";

	return 0;
}
```

#### 4.用C++运行Torch Script

```c++
#include<torch/script.h>
#include <torch/torch.h>
#include<iostream>
#include<memory>

int main() {
	torch::jit::script::Module module;

	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load("./traced_resnet_model.pt");
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	std::cout << "ok\n";

	// Create a vector of inputs.
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(torch::ones({ 1,3,512,512 }));

	// Execute the model and turn its output into a tensor.
	at::Tensor output = module.forward(inputs).toTensor();
	std::cout << output.slice(1, 0, 5) << std::endl;

	return 0;
}
```

