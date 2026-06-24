from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)
(ROOT / ".cache").mkdir(exist_ok=True)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def rel(path: Path, start: Path) -> str:
    return path.resolve().relative_to(start.resolve()).as_posix()


def make_html(
    output_path: Path,
    artifact_dir: Path,
    summary: dict,
    metrics: dict,
    search_rows: list[dict[str, str]],
    github_url: str,
    weights_url: str,
) -> None:
    fig_dir = artifact_dir / "figures"
    loss_img = rel(fig_dir / "loss_curves.png", output_path.parent)
    acc_img = rel(fig_dir / "accuracy_curves.png", output_path.parent)
    cm_img = rel(fig_dir / "confusion_matrix.png", output_path.parent)
    weights_img = rel(fig_dir / "first_layer_weights.png", output_path.parent)
    error_img = rel(fig_dir / "error_examples.png", output_path.parent)

    best_search = None
    if search_rows:
        best_search = max(search_rows, key=lambda r: float(r.get("best_val_acc", 0.0)))

    needs_link_update = "TODO" in github_url or "TODO" in weights_url
    todo_block = (
        '<div class="todo">注意：如果上述仍是 TODO，请先把本目录上传到你的 Public GitHub Repo，并把 '
        '<code>artifacts/best_model.npz</code> 上传到 Google Drive，再用真实链接重新生成报告。</div>'
        if needs_link_update
        else ""
    )

    search_table = ""
    if search_rows:
        search_table += "<table><thead><tr>"
        headers = ["run", "hidden_dim", "activation", "lr", "weight_decay", "best_val_acc", "final_train_acc"]
        search_table += "".join(f"<th>{h}</th>" for h in headers)
        search_table += "</tr></thead><tbody>"
        for row in search_rows:
            search_table += "<tr>" + "".join(
                f"<td>{html.escape(fmt_pct(row[h]) if h.endswith('acc') else str(row[h]))}</td>" for h in headers
            ) + "</tr>"
        search_table += "</tbody></table>"

    body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HW1 EuroSAT 三层神经网络分类器实验报告</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #5b6475;
      --line: #d9dee8;
      --panel: #f6f8fb;
      --accent: #166b5b;
      --accent-2: #b25534;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      line-height: 1.65;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      width: min(1080px, calc(100% - 40px));
      margin: 0 auto;
      padding: 42px 0 64px;
    }}
    header {{
      border-bottom: 2px solid var(--line);
      padding-bottom: 22px;
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin-top: 32px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
      font-size: 22px;
      letter-spacing: 0;
    }}
    h3 {{ margin: 20px 0 8px; font-size: 17px; }}
    p {{ margin: 9px 0; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .metric {{
      background: var(--panel);
      border-left: 4px solid var(--accent);
      padding: 10px 12px;
    }}
    .metric strong {{ display: block; font-size: 20px; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #eef1f5;
      padding: 1px 5px;
      border-radius: 4px;
    }}
    pre {{
      background: #111827;
      color: #e5e7eb;
      padding: 14px 16px;
      overflow-x: auto;
      border-radius: 6px;
      line-height: 1.45;
    }}
    figure {{
      margin: 18px 0;
      border: 1px solid var(--line);
      padding: 12px;
      background: #ffffff;
    }}
    figure img {{ width: 100%; display: block; }}
    figcaption {{ color: var(--muted); font-size: 14px; margin-top: 8px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin: 12px 0 18px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
    }}
    th {{ background: var(--panel); }}
    .todo {{
      border-left: 4px solid var(--accent-2);
      background: #fff6f1;
      padding: 10px 12px;
      margin: 14px 0;
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>HW1：从零开始构建三层神经网络分类器，实现地表覆盖图像分类</h1>
    <p>本报告使用 NumPy 手工实现三层 MLP，在 EuroSAT RGB 数据集上完成土地覆盖分类。模型、损失函数、SGD、学习率衰减、L2 正则化、反向传播、超参数搜索、测试评估和可视化均由本项目代码完成。</p>
    <div class="meta">
      <div class="metric"><span>验证集最佳准确率</span><strong>{fmt_pct(summary.get("best_val_acc"))}</strong></div>
      <div class="metric"><span>测试集准确率</span><strong>{fmt_pct(metrics.get("test_accuracy"))}</strong></div>
      <div class="metric"><span>输入尺寸</span><strong>{summary.get("image_size", "N/A")} x {summary.get("image_size", "N/A")} RGB</strong></div>
      <div class="metric"><span>隐藏层维度</span><strong>{summary.get("hidden_dim", "N/A")}</strong></div>
    </div>
  </header>

  <section>
    <h2>1. 代码与提交链接</h2>
    <p>GitHub Repo：<a href="{html.escape(github_url)}">{html.escape(github_url)}</a></p>
    <p>模型权重下载地址：<a href="{html.escape(weights_url)}">{html.escape(weights_url)}</a></p>
    {todo_block}
  </section>

  <section>
    <h2>2. 数据集处理</h2>
    <p>EuroSAT_RGB 共包含 10 类遥感图像。本实验按类别进行分层切分：70% 训练集、15% 验证集、15% 测试集。图像统一缩放为 {summary.get("image_size", "N/A")} x {summary.get("image_size", "N/A")}，转换为 RGB 后展平为向量，并使用训练集均值和标准差进行标准化。</p>
    <p>样本数：训练集 {summary.get("n_train", "N/A")}，验证集 {summary.get("n_val", "N/A")}，测试集 {summary.get("n_test", "N/A")}。</p>
  </section>

  <section>
    <h2>3. 模型与优化方法</h2>
    <p>模型结构为输入层、一个隐藏层和输出层：<code>x -> Linear -> {html.escape(str(summary.get("activation", "relu")))} -> Linear -> Softmax</code>。交叉熵损失使用稳定 Softmax 实现，Linear 层和激活函数都保存前向传播缓存并手工实现 <code>backward</code>。训练时使用 SGD 更新参数，并加入学习率指数衰减与 L2 正则化。</p>
    <p>最终训练超参数：初始学习率 {summary.get("lr", "N/A")}，学习率衰减 {summary.get("lr_decay", "N/A")}，Weight Decay {summary.get("weight_decay", "N/A")}，Batch Size {summary.get("batch_size", "N/A")}，训练轮数 {summary.get("epochs", "N/A")}。最佳模型根据验证集准确率自动保存，最佳轮次为第 {summary.get("best_epoch", "N/A")} 轮。</p>
  </section>

  <section>
    <h2>4. 超参数搜索</h2>
    <p>搜索阶段对隐藏层维度、激活函数、学习率和正则化强度进行网格搜索。为了控制时间成本，搜索使用每类样本上限；最终模型再使用完整设定训练。</p>
    {search_table}
    <p>最优组合：{html.escape(str(best_search)) if best_search else "N/A"}</p>
  </section>

  <section>
    <h2>5. 训练过程可视化</h2>
    <figure><img src="{loss_img}" alt="Loss curves"><figcaption>训练集与验证集交叉熵 Loss 曲线。</figcaption></figure>
    <figure><img src="{acc_img}" alt="Accuracy curves"><figcaption>训练集和验证集 Accuracy 曲线。验证集曲线用于选择最佳模型权重。</figcaption></figure>
  </section>

  <section>
    <h2>6. 测试集结果</h2>
    <p>最佳模型在独立测试集上的 Accuracy 为 {fmt_pct(metrics.get("test_accuracy"))}。混淆矩阵如下，行是真实类别，列是预测类别。</p>
    <figure><img src="{cm_img}" alt="Confusion matrix"><figcaption>测试集混淆矩阵。</figcaption></figure>
  </section>

  <section>
    <h2>7. 第一层权重可视化与空间模式观察</h2>
    <p>将第一层权重矩阵的每个隐藏单元列向量恢复为 {summary.get("image_size", "N/A")} x {summary.get("image_size", "N/A")} x 3 的 RGB 图像后，可以看到权重图整体较为高频、碎片化，只有少数隐藏单元表现出弱的绿色、蓝色或灰白颜色偏好。它们可能分别帮助模型捕捉植被、水体或建筑区域的全局颜色差异，但没有形成非常清晰、稳定的 “河流” 或 “森林” 局部空间模板。这与 MLP 将图像展平后再分类有关：模型缺少卷积网络的局部连接和平移共享先验，因此第一层权重更像点状颜色响应，而不是可解释的边缘或纹理检测器。</p>
    <figure><img src="{weights_img}" alt="First layer weights"><figcaption>第一层隐藏单元权重图。</figcaption></figure>
  </section>

  <section>
    <h2>8. 错例分析</h2>
    <p>错例主要集中在视觉纹理或颜色接近的类别之间：Highway 与 River 都可能出现细长、连续、低纹理的线状结构；Pasture、AnnualCrop、PermanentCrop 与 HerbaceousVegetation 都包含大面积绿色或黄绿色植被纹理；Industrial 与 Residential 都有规则建筑块和道路网。MLP 只能基于展平后的全局像素模式分类，因此遇到尺度、方向和局部布局变化时更容易混淆。</p>
    <figure><img src="{error_img}" alt="Error examples"><figcaption>测试集中若干分类错误样例，标题中 T 表示真实类别，P 表示预测类别。</figcaption></figure>
  </section>

  <section>
    <h2>9. 运行方式</h2>
    <pre><code>python3 search.py --data-dir /Users/cheese/Downloads/hw1/EuroSAT_RGB
python3 train.py --data-dir /Users/cheese/Downloads/hw1/EuroSAT_RGB
python3 evaluate.py --data-dir /Users/cheese/Downloads/hw1/EuroSAT_RGB
python3 visualize.py --data-dir /Users/cheese/Downloads/hw1/EuroSAT_RGB
python3 make_report.py --github-url {html.escape(github_url)} --weights-url {html.escape(weights_url)}</code></pre>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(body, encoding="utf-8")


def make_pdf(
    output_path: Path,
    artifact_dir: Path,
    summary: dict,
    metrics: dict,
    search_rows: list[dict[str, str]],
    github_url: str,
    weights_url: str,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from PIL import Image as PILImage
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "ChineseBody",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=15,
        spaceAfter=7,
    )
    h1 = ParagraphStyle("ChineseH1", parent=base, fontSize=18, leading=23, spaceAfter=12)
    h2 = ParagraphStyle("ChineseH2", parent=base, fontSize=14, leading=18, spaceBefore=12, spaceAfter=8)
    small = ParagraphStyle("ChineseSmall", parent=base, fontSize=8.5, leading=11)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
    )
    story = []
    needs_link_update = "TODO" in github_url or "TODO" in weights_url

    def p(text: str, style=base):
        story.append(Paragraph(text, style))

    def img(name: str, width: float = 6.6, max_height: float = 4.7):
        path = artifact_dir / "figures" / name
        if path.exists():
            with PILImage.open(path) as im:
                w_px, h_px = im.size
            draw_w = width * inch
            draw_h = draw_w * h_px / max(w_px, 1)
            max_h = max_height * inch
            if draw_h > max_h:
                scale = max_h / draw_h
                draw_w *= scale
                draw_h *= scale
            story.append(Image(str(path), width=draw_w, height=draw_h))
            story.append(Spacer(1, 8))

    p("HW1：从零开始构建三层神经网络分类器，实现地表覆盖图像分类", h1)
    p(
        f"本报告使用 NumPy 手工实现三层 MLP，在 EuroSAT RGB 数据集上完成土地覆盖分类。"
        f"最佳验证集准确率为 {fmt_pct(summary.get('best_val_acc'))}，测试集准确率为 {fmt_pct(metrics.get('test_accuracy'))}。"
    )
    p(f"GitHub Repo：{github_url}")
    p(f"模型权重下载地址：{weights_url}")

    p("1. 数据集处理", h2)
    p(
        f"数据集按类别进行 70%/15%/15% 分层切分。图像缩放为 {summary.get('image_size', 'N/A')} x "
        f"{summary.get('image_size', 'N/A')} RGB 后展平，并使用训练集均值和标准差标准化。"
        f"样本数：训练 {summary.get('n_train', 'N/A')}，验证 {summary.get('n_val', 'N/A')}，测试 {summary.get('n_test', 'N/A')}。"
    )

    p("2. 模型与优化", h2)
    p(
        f"模型为 Input -> Linear -> {summary.get('activation', 'relu')} -> Linear -> Softmax。"
        f"Linear 层、激活函数和 Cross-Entropy Loss 均保存前向缓存并手工实现 backward。"
        f"训练采用 SGD、学习率指数衰减和 L2 正则化。隐藏层维度 {summary.get('hidden_dim', 'N/A')}，"
        f"初始学习率 {summary.get('lr', 'N/A')}，衰减 {summary.get('lr_decay', 'N/A')}，"
        f"Weight Decay {summary.get('weight_decay', 'N/A')}，最佳轮次 {summary.get('best_epoch', 'N/A')}。"
    )

    p("3. 超参数搜索", h2)
    if search_rows:
        table_data = [["run", "hidden", "act", "lr", "wd", "val acc"]]
        for row in search_rows:
            table_data.append(
                [
                    row.get("run", ""),
                    row.get("hidden_dim", ""),
                    row.get("activation", ""),
                    row.get("lr", ""),
                    row.get("weight_decay", ""),
                    fmt_pct(row.get("best_val_acc")),
                ]
            )
        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd6e3")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))
    else:
        p("未发现超参数搜索表。")

    p("4. 训练过程可视化", h2)
    img("loss_curves.png")
    img("accuracy_curves.png")

    p("5. 测试集结果", h2)
    p(f"最佳模型在独立测试集上的 Accuracy 为 {fmt_pct(metrics.get('test_accuracy'))}。混淆矩阵行是真实类别，列是预测类别。")
    img("confusion_matrix.png", width=6.25)

    p("6. 第一层权重可视化与空间模式观察", h2)
    p(
        "第一层权重恢复为 RGB 图像后，整体较为高频、碎片化，只能看到少数隐藏单元存在弱的绿色、蓝色或灰白颜色偏好。"
        "这些偏好可能帮助模型区分植被、水体或建筑区域的全局颜色差异，但没有形成非常清晰、稳定的河流或森林局部空间模板。"
        "这与 MLP 展平图像后再分类有关：模型缺少卷积网络的局部连接和平移共享先验，"
        "因此第一层权重更像点状颜色响应，而不是可解释的边缘或纹理检测器。"
    )
    img("first_layer_weights.png")

    p("7. 错例分析", h2)
    p(
        "错例多出现在视觉相近的类别之间：Highway 与 River 都可能呈现细长线状结构；"
        "Pasture、AnnualCrop、PermanentCrop 与 HerbaceousVegetation 都含有大面积绿色或黄绿色纹理；"
        "Industrial 与 Residential 都包含规则建筑块和道路网。"
    )
    img("error_examples.png")

    p("8. 运行方式", h2)
    if needs_link_update:
        p(
            "依次运行 search.py、train.py、evaluate.py、visualize.py 和 make_report.py。"
            "真实提交前，将 GitHub Repo 和模型权重地址替换为公开链接。", small
        )
    else:
        p(
            "依次运行 search.py、train.py、evaluate.py、visualize.py 和 make_report.py。"
            "本报告已包含公开 GitHub Repo 和模型权重下载链接。", small
        )

    doc.build(story)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HTML and PDF reports from generated artifacts.")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--html-out", type=Path, default=ROOT / "report.html")
    parser.add_argument("--pdf-out", type=Path, default=ROOT / "report.pdf")
    parser.add_argument("--github-url", type=str, default="TODO: replace with your public GitHub repo URL")
    parser.add_argument("--weights-url", type=str, default="TODO: replace with your Google Drive model weight URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_json(args.artifact_dir / "training_summary.json", {})
    metrics = load_json(args.artifact_dir / "test_metrics.json", {})
    search_rows = load_csv(args.artifact_dir / "search" / "search_results.csv")
    make_html(args.html_out, args.artifact_dir, summary, metrics, search_rows, args.github_url, args.weights_url)
    make_pdf(args.pdf_out, args.artifact_dir, summary, metrics, search_rows, args.github_url, args.weights_url)
    print(f"Wrote {args.html_out}")
    print(f"Wrote {args.pdf_out}")


if __name__ == "__main__":
    main()
