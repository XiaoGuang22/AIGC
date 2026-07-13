import os
import re


def cleanup_aborted_logs(log_dir):
    # 使用正则表达式匹配，\d+ 代表匹配一个或多个数字
    # 匹配模式：>>> Start Training for [数字] Epochs...
    pattern = re.compile(r">>> Start Training for \d+ Epochs\.\.\.")
    deleted_count = 0

    if not os.path.exists(log_dir):
        print(f"❌ 错误：目录 '{log_dir}' 不存在。")
        return

    print(f"�� 正在扫描目录: {log_dir}")
    print("-" * 40)

    for filename in os.listdir(log_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(log_dir, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 读取所有行，过滤掉纯空行
                    content = [line.strip() for line in f if line.strip()]

                if not content:
                    continue

                # 获取最后一行有效内容
                last_line = content[-1]

                # 如果最后一行匹配启动标志，说明该实验瞬间夭折
                if pattern.fullmatch(last_line):
                    print(f"��️  删除无效日志: {filename}")
                    os.remove(file_path)
                    deleted_count += 1

            except Exception as e:
                print(f"⚠️  处理 {filename} 时出错: {e}")

    print("-" * 40)
    print(f"✅ 清理完成！共删除 {deleted_count} 个无效日志文件。")


if __name__ == "__main__":
    # 设定你的日志路径
    TARGET_DIR = "../results/logs"
    cleanup_aborted_logs(TARGET_DIR)