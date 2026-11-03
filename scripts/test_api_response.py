"""
测试 API 响应格式
只处理第一条数据，用于调试 API 响应结构
"""
import json
from openai import OpenAI
import traceback


def test_api_response():
    """测试 API 响应格式"""
    
    # 测试数据
    test_data = {
        "name": "辉钼矿",
        "english_name": "Molybdenite",
        "main_purpose": "工业原料；科学研究；观赏",
        "brief_characterization": "铅灰色，呈磷片状产出，条痕亮浅灰色，金属光泽，不透明，平行{0001}极完全解理，硬度1，有滑感，化学式为MoS₂",
        "specific_purpose": "钼、铼的矿石矿物。"
    }
    
    # 创建客户端
    client = OpenAI(
        base_url="https://api-inference.modelscope.cn/v1",
        api_key="ms-ms-dfd7ef08-e098-42f7-a210-cd0d78fc40f0"
    )
    
    # 构建提示词
    prompt = f"""请根据以下矿物信息，将主要用途、简要特征、具体用途合并成一段连贯的自然语言描述：

矿物名称：{test_data['name']}
英文名称：{test_data['english_name']}
主要用途：{test_data['main_purpose']}
简要特征：{test_data['brief_characterization']}
具体用途：{test_data['specific_purpose']}

要求：
- 将主要用途、简要特征、具体用途合并成一段连贯的自然语言描述
- 使用专业但易懂的语言
- 保持信息的完整性和准确性
- 形成流畅的段落，避免简单罗列
- 描述要详细，充分展开每个方面的信息

请直接输出描述内容，不要包含任何标题或格式标记。"""

    print("=" * 60)
    print("测试 API 响应格式")
    print("=" * 60)
    
    # 测试不同的模型和参数
    test_cases = [
        {
            "name": "原始模型",
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 2000
        },
        {
            "name": "不使用 stream",
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 500
        },
        {
            "name": "使用 stream=True",
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2000
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"测试: {test_case['name']}")
        print(f"模型: {test_case['model']}")
        print(f"Stream: {test_case['stream']}")
        print(f"{'='*60}")
        
        try:
            # 调用 API
            print("调用 API...")
            
            if test_case['stream']:
                # 流式响应
                response = client.chat.completions.create(
                    model=test_case['model'],
                    messages=[
                        {
                            'role': 'system',
                            'content': '你是一个专业的矿物学专家，擅长将矿物数据改写为自然语言描述。'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    stream=test_case['stream'],
                    temperature=test_case['temperature'],
                    max_tokens=test_case['max_tokens']
                )
                
                print("\n✅ 流式 API 调用成功!")
                print("接收流式响应...")
                
                full_content = ""
                for chunk in response:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            full_content += delta.content
                            print(delta.content, end='', flush=True)
                
                print(f"\n\n完整内容长度: {len(full_content)}")
                
                if full_content:
                    result = {
                        'name': test_data['name'],
                        'English_name': test_data['english_name'],
                        '描述': full_content
                    }
                    print(f"\n最终结果:")
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print("❌ 流式响应内容为空")
                    
            else:
                # 非流式响应
                response = client.chat.completions.create(
                    model=test_case['model'],
                    messages=[
                        {
                            'role': 'system',
                            'content': '你是一个专业的矿物学专家，擅长将矿物数据改写为自然语言描述。'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    stream=test_case['stream'],
                    temperature=test_case['temperature'],
                    max_tokens=test_case['max_tokens']
                )
                
                print("\n✅ API 调用成功!")
                print(f"\n响应类型: {type(response)}")
                print(f"响应属性: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                
                # 检查响应内容
                print(f"\n响应内容: {response}")
                
                # 尝试获取结果
                if hasattr(response, 'choices'):
                    print(f"\n✅ 有 choices 字段")
                    print(f"Choices 数量: {len(response.choices) if response.choices else 0}")
                    
                    if response.choices:
                        choice = response.choices[0]
                        print(f"第一个 choice 类型: {type(choice)}")
                        print(f"第一个 choice 属性: {[attr for attr in dir(choice) if not attr.startswith('_')]}")
                        
                        if hasattr(choice, 'message'):
                            print(f"\n✅ 有 message 字段")
                            message = choice.message
                            print(f"Message 类型: {type(message)}")
                            print(f"Message 属性: {[attr for attr in dir(message) if not attr.startswith('_')]}")
                            
                            if hasattr(message, 'content'):
                                print(f"\n✅ 有 content 字段")
                                content = message.content
                                print(f"Content 长度: {len(content) if content else 0}")
                                print(f"\n生成的描述:\n{content}")
                                
                                # 构建结果
                                result = {
                                    'name': test_data['name'],
                                    'English_name': test_data['english_name'],
                                    '描述': content
                                }
                                
                                print(f"\n最终结果:")
                                print(json.dumps(result, ensure_ascii=False, indent=2))
                                
                            else:
                                print(f"❌ Message 对象缺少 content 字段")
                        else:
                            print(f"❌ Choice 对象缺少 message 字段")
                    else:
                        print(f"❌ Choices 列表为空")
                        print("这可能是 API key 权限问题或模型配置问题")
                else:
                    print(f"❌ 响应缺少 choices 字段")
                    print(f"响应的完整结构: {response}")
                    
        except Exception as e:
            print(f"\n❌ API 调用失败: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    test_api_response()