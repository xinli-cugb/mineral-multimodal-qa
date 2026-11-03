"""
测试矿物数据重构脚本
演示如何使用重构后的数据格式
"""
import json
from pathlib import Path
from refactor_mineral_data import MineralDataRefactor


def test_refactor_single():
    """测试单条数据重构"""
    print("=" * 60)
    print("测试单条数据重构")
    print("=" * 60)
    
    # 测试数据
    test_data = {
        "name": "辉钼矿",
        "english_name": "Molybdenite",
        "place_of_origin": "不详",
        "place_of_province": "不详",
        "place_of_country": "不详",
        "the_provider": "张平",
        "the_storage_location": "新疆地质矿产博物馆",
        "photo_url": "http://example.com/photo.jpg",
        "link": "http://example.com/link",
        "main_purpose": "工业原料；科学研究；观赏",
        "brief_characterization": "铅灰色，呈磷片状产出，条痕亮浅灰色，金属光泽，不透明，平行{0001}极完全解理，硬度1，有滑感，化学式为MoS₂",
        "specific_purpose": "钼、铼的矿石矿物。"
    }
    
    # 创建重构器（需要替换为真实的 API Key）
    try:
        refactor = MineralDataRefactor(
            api_key="your_api_key_here"  # 替换为真实的 API Key
        )
        
        # 重构数据
        result = refactor.refactor_mineral_data(test_data)
        
        # 输出结果
        print("\n重构结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 验证格式
        assert 'name' in result, "缺少 name 字段"
        assert 'English_name' in result, "缺少 English_name 字段"
        assert '描述' in result, "缺少 描述 字段"
        
        print("\n✅ 格式验证通过!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("提示: 请替换为真实的 ModelScope API Key")


def test_refactor_batch():
    """测试批量数据重构"""
    print("\n" + "=" * 60)
    print("测试批量数据重构")
    print("=" * 60)
    
    # 读取测试数据
    test_file = "/root/V2_0_Mineral_QA/text_data/all_v1.json"
    
    if not Path(test_file).exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是单条记录，转换为列表
    if isinstance(data, dict):
        data = [data]
    
    print(f"共 {len(data)} 条数据")
    
    # 限制处理数量（用于测试）
    test_data = data[:3]  # 只处理前3条
    print(f"测试处理前 {len(test_data)} 条数据")
    
    # 创建重构器
    try:
        refactor = MineralDataRefactor(
            api_key="your_api_key_here"  # 替换为真实的 API Key
        )
        
        # 批量重构
        output_file = "/root/V2_0_Mineral_QA/text_data/test_refactored.json"
        refactor.refactor_batch(
            data_list=test_data,
            output_file=output_file,
            batch_size=1,
            delay=0.5
        )
        
        # 验证输出文件
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"\n✅ 批量重构完成!")
        print(f"输出文件: {output_file}")
        print(f"处理数量: {len(results)}")
        
        # 显示第一条结果
        if results:
            print("\n第一条重构结果:")
            print(json.dumps(results[0], ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("提示: 请替换为真实的 ModelScope API Key")


def show_expected_format():
    """显示预期的输出格式"""
    print("\n" + "=" * 60)
    print("预期的输出格式")
    print("=" * 60)
    
    example = {
        "name": "辉钼矿",
        "English_name": "Molybdenite",
        "描述": "辉钼矿是一种重要的工业原料，广泛应用于科学研究、观赏和工业生产领域。它具有铅灰色、呈磷片状产出的特征，条痕为亮浅灰色，具有金属光泽，不透明。该矿物具有平行{0001}极完全解理，硬度为1，触摸时有滑感，化学式为MoS₂。作为钼、铼的重要矿石矿物，辉钼矿在冶金工业和化学工业中具有重要价值，是提取钼和铼元素的主要来源。"
    }
    
    print("\n示例输出:")
    print(json.dumps(example, ensure_ascii=False, indent=2))
    
    print("\n字段说明:")
    print("- name: 矿物中文名称")
    print("- English_name: 矿物英文名称")
    print("- 描述: 将主要用途、简要特征、具体用途合并成的连贯自然语言描述")


def main():
    """主函数"""
    print("\n矿物数据重构脚本测试")
    print("=" * 60)
    
    # 显示预期格式
    show_expected_format()
    
    # 测试单条数据重构
    # test_refactor_single()
    
    # 测试批量数据重构
    # test_refactor_batch()
    
    print("\n" + "=" * 60)
    print("测试说明:")
    print("1. 需要替换为真实的 ModelScope API Key")
    print("2. 取消注释相应的测试函数来运行测试")
    print("3. 批量测试会处理前3条数据")
    print("=" * 60)


if __name__ == "__main__":
    main()