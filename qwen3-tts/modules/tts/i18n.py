"""Interface strings (vi/en/zh) and the model/mode choice builders."""

from __future__ import annotations

from .config import FISH_LABEL


DEFAULT_LANG = "en"
LANG_CHOICES = [("Tiếng Việt", "vi"), ("English", "en"), ("中文", "zh")]

T = {
    "en": {
        "tab_single": "Single",
        "tab_batch": "Batch",
        "sec_text": "✍️ **Text**",
        "text_label": "Chinese text",
        "text_ph": "在这里输入中文…",
        "sec_voice": "🎤 **Voice & Model**",
        "model": "Model",
        "model_hq": "Higher quality — 1.7B",
        "model_clone": "Voice clone — Fish S2 Pro",
        "ref_audio": "Reference voice clip (5–15s)",
        "ref_text": "Reference transcript (optional)",
        "ref_text_ph": "Type what is said in the reference clip — improves cloning",
        "clone_note": (
            "🎭 Speaks your text in the uploaded voice. Leave the transcript blank "
            "and Whisper fills it in. `[happy]` / `[whisper]` tags: **Fish S2 Pro** only."
        ),
        "saved_head": "💼 **Saved voices**",
        "saved_pick": "Saved voices",
        "save_as": "Save the clip above as…",
        "save_as_ph": "e.g. Teacher Li",
        "save_voice": "💾 Save",
        "load_voice": "⬆️ Load",
        "saved_hint": "Save the uploaded clip + transcript to reuse this voice later without re-uploading.",
        "saved_ok": "✅ Saved “{name}” — reload it any time from the dropdown.",
        "loaded_ok": "✅ Loaded “{name}”.",
        "deleted_ok": "🗑 Deleted “{name}”.",
        "pick_first": "Pick a saved voice from the dropdown first.",
        "voice": "Voice",
        "sec_pron": "🔊 **Pronunciation**",
        "mode": "Reading mode",
        "mode_word": "Single word",
        "mode_phrase": "Phrase",
        "mode_sent": "Sentence",
        "mode_para": "Paragraph",
        "speed": "Speed (×)",
        "style": "Style instruction (optional)",
        "style_ph": "例如：像中文老师一样，慢速清晰地朗读",
        "acc_examples": "Style examples",
        "p_before": "Pause before (s)",
        "p_after": "Pause after (s)",
        "p_between": "Pause between repeats (s)",
        "p_sentence": "Pause after 。！？ (s)",
        "p_comma": "Pause after ，、 (s)",
        "p_paragraph": "Pause at line break (s)",
        "pause_hint": (
            "Real silence is inserted at those marks. Any value above 0 also makes "
            "that sentence its own take — a cleaner break, slightly less flowing "
            "narration. Set all three to 0 for one continuous reading."
        ),
        "repeat": "Repeat count",
        "sec_out": "💾 **Output**",
        "format": "Format",
        "quality": "MP3 quality",
        "generate": "🎧 Generate audio",
        "audio": "Preview",
        "download": "⬇️ Download audio",
        "batch_desc": (
            "Paste **one Chinese word or sentence per line**, or upload a CSV with "
            "columns `text, voice, model, mode, speed` (only `text` is required). "
            "Rows use the defaults on the right."
        ),
        "batch_text": "One item per line",
        "batch_csv": "Or upload CSV",
        "b_defaults": "⚙️ **Defaults for all rows**",
        "b_style": "Style instruction for all rows (optional)",
        "acc_adv": "Pauses & repeats",
        "batch_btn": "🚀 Generate batch",
        "table": "Results",
        "zip": "⬇️ Download ZIP (audio + results CSV)",
        "footer": "Files are saved in",
        "tab_files": "Library",
        "files_hint": "Files in the outputs folder, newest first — click a row to play audio or preview text.",
        "refresh": "🔄 Refresh",
        "open_folder": "📂 Open folder",
        "files_table": "Output files",
        "file_player": "Play selected file",
        "file_text": "File contents",
        "res_cached": "♻️ From cache",
        "res_new": "🆕 Newly generated",
        "res_file": "Saved file",
        "loading": "Loading model: {model} … this can take a moment.",
        "item_of": "Item {n} of {total}",
        "packaging": "Packaging results",
        "batch_done": "**Done.** {ok} succeeded, {failed} failed, {total} total.",
    },
    "vi": {
        "tab_single": "Tạo đơn",
        "tab_batch": "Tạo hàng loạt",
        "sec_text": "✍️ **Văn bản**",
        "text_label": "Văn bản tiếng Trung",
        "text_ph": "在这里输入中文…",
        "sec_voice": "🎤 **Giọng & Mô hình**",
        "model": "Mô hình",
        "model_hq": "Chất lượng cao — 1.7B",
        "model_clone": "Nhân bản giọng — Fish S2 Pro",
        "ref_audio": "Mẫu giọng tham chiếu (5–15 giây)",
        "ref_text": "Lời thoại của mẫu (tuỳ chọn)",
        "ref_text_ph": "Nhập nội dung nói trong mẫu — giúp nhân bản chính xác hơn",
        "clone_note": (
            "🎭 Đọc văn bản bằng giọng đã tải lên. Bỏ trống lời thoại thì Whisper tự "
            "điền. Thẻ `[happy]` / `[whisper]`: chỉ **Fish S2 Pro**."
        ),
        "saved_head": "💼 **Giọng đã lưu**",
        "saved_pick": "Giọng đã lưu",
        "save_as": "Lưu mẫu ở trên thành…",
        "save_as_ph": "ví dụ: Cô Lan",
        "save_voice": "💾 Lưu",
        "load_voice": "⬆️ Nạp",
        "saved_hint": "Lưu mẫu giọng + lời thoại để dùng lại lần sau mà không cần tải lên lại.",
        "saved_ok": "✅ Đã lưu “{name}” — nạp lại bất cứ lúc nào từ danh sách.",
        "loaded_ok": "✅ Đã nạp “{name}”.",
        "deleted_ok": "🗑 Đã xoá “{name}”.",
        "pick_first": "Hãy chọn một giọng đã lưu từ danh sách trước.",
        "voice": "Giọng đọc",
        "sec_pron": "🔊 **Phát âm**",
        "mode": "Chế độ đọc",
        "mode_word": "Từ đơn",
        "mode_phrase": "Cụm từ",
        "mode_sent": "Câu",
        "mode_para": "Đoạn văn",
        "speed": "Tốc độ (×)",
        "style": "Chỉ dẫn phong cách (tuỳ chọn)",
        "style_ph": "例如：像中文老师一样，慢速清晰地朗读",
        "acc_examples": "Ví dụ phong cách",
        "p_before": "Nghỉ trước (giây)",
        "p_after": "Nghỉ sau (giây)",
        "p_between": "Nghỉ giữa các lần lặp (giây)",
        "p_sentence": "Nghỉ sau 。！？ (giây)",
        "p_comma": "Nghỉ sau ，、 (giây)",
        "p_paragraph": "Nghỉ khi xuống dòng (giây)",
        "pause_hint": (
            "Chèn khoảng lặng thật tại các dấu này. Giá trị lớn hơn 0 cũng khiến mỗi "
            "câu được tạo riêng — tách câu rõ hơn, giọng đọc bớt liền mạch một chút. "
            "Đặt cả ba về 0 để đọc liền một mạch."
        ),
        "repeat": "Số lần lặp",
        "sec_out": "💾 **Đầu ra**",
        "format": "Định dạng",
        "quality": "Chất lượng MP3",
        "generate": "🎧 Tạo âm thanh",
        "audio": "Nghe thử",
        "download": "⬇️ Tải tệp âm thanh",
        "batch_desc": (
            "Dán **mỗi dòng một từ hoặc một câu tiếng Trung**, hoặc tải lên tệp CSV "
            "có các cột `text, voice, model, mode, speed` (chỉ bắt buộc cột `text`). "
            "Các dòng dùng cài đặt mặc định ở bên phải."
        ),
        "batch_text": "Mỗi dòng một mục",
        "batch_csv": "Hoặc tải lên CSV",
        "b_defaults": "⚙️ **Cài đặt mặc định cho mọi dòng**",
        "b_style": "Chỉ dẫn phong cách cho tất cả các dòng (tuỳ chọn)",
        "acc_adv": "Nghỉ & lặp lại",
        "batch_btn": "🚀 Tạo hàng loạt",
        "table": "Kết quả",
        "zip": "⬇️ Tải ZIP (âm thanh + CSV kết quả)",
        "footer": "Tệp được lưu tại",
        "tab_files": "Thư viện",
        "files_hint": "Tệp trong thư mục đầu ra, mới nhất trước — nhấp vào một dòng để phát âm thanh hoặc xem nội dung.",
        "refresh": "🔄 Làm mới",
        "open_folder": "📂 Mở thư mục",
        "files_table": "Tệp đầu ra",
        "file_player": "Phát tệp đã chọn",
        "file_text": "Nội dung tệp",
        "res_cached": "♻️ Lấy từ bộ nhớ đệm",
        "res_new": "🆕 Vừa tạo mới",
        "res_file": "Tệp đã lưu",
        "loading": "Đang nạp mô hình: {model}…",
        "item_of": "Mục {n} / {total}",
        "packaging": "Đang đóng gói kết quả",
        "batch_done": "**Hoàn tất.** {ok} thành công, {failed} thất bại, tổng cộng {total}.",
    },
    "zh": {
        "tab_single": "单条生成",
        "tab_batch": "批量生成",
        "sec_text": "✍️ **文本**",
        "text_label": "中文文本",
        "text_ph": "在这里输入中文…",
        "sec_voice": "🎤 **声音与模型**",
        "model": "模型",
        "model_hq": "高质量 — 1.7B",
        "model_clone": "声音克隆 — Fish S2 Pro",
        "ref_audio": "参考声音片段（5–15 秒）",
        "ref_text": "参考文本（可选）",
        "ref_text_ph": "输入参考片段中所说的内容 — 可提升克隆效果",
        "clone_note": (
            "🎭 用上传的声音朗读文本。参考文本留空则由 Whisper 自动转写。"
            "`[happy]`、`[whisper]` 标签仅适用于 **Fish S2 Pro**。"
        ),
        "saved_head": "💼 **已保存的声音**",
        "saved_pick": "已保存的声音",
        "save_as": "将上方片段保存为…",
        "save_as_ph": "例如：李老师",
        "save_voice": "💾 保存",
        "load_voice": "⬆️ 载入",
        "saved_hint": "保存上传的声音片段和参考文本，下次无需重新上传即可复用。",
        "saved_ok": "✅ 已保存“{name}”— 随时可从下拉菜单重新载入。",
        "loaded_ok": "✅ 已载入“{name}”。",
        "deleted_ok": "🗑 已删除“{name}”。",
        "pick_first": "请先从下拉菜单中选择一个已保存的声音。",
        "voice": "声音",
        "sec_pron": "🔊 **发音设置**",
        "mode": "朗读模式",
        "mode_word": "单词",
        "mode_phrase": "词组",
        "mode_sent": "句子",
        "mode_para": "段落",
        "speed": "语速 (×)",
        "style": "风格指令（可选）",
        "style_ph": "例如：像中文老师一样，慢速清晰地朗读",
        "acc_examples": "风格示例",
        "p_before": "前停顿（秒）",
        "p_after": "后停顿（秒）",
        "p_between": "重复间停顿（秒）",
        "p_sentence": "句末停顿（秒）",
        "p_comma": "逗号停顿（秒）",
        "p_paragraph": "换行停顿（秒）",
        "pause_hint": (
            "在这些标点处插入真正的静音。数值大于 0 时，每句话会单独合成——"
            "断句更清楚，但连贯感略降。三项都设为 0 则一气呵成地朗读。"
        ),
        "repeat": "重复次数",
        "sec_out": "💾 **输出**",
        "format": "格式",
        "quality": "MP3 音质",
        "generate": "🎧 生成语音",
        "audio": "试听",
        "download": "⬇️ 下载音频",
        "batch_desc": (
            "每行粘贴**一个中文词语或句子**，或上传包含 "
            "`text, voice, model, mode, speed` 列的 CSV（仅 `text` 必填）。"
            "各行以右侧设置为默认值。"
        ),
        "batch_text": "每行一条",
        "batch_csv": "或上传 CSV",
        "b_defaults": "⚙️ **所有行的默认设置**",
        "b_style": "应用于所有行的风格指令（可选）",
        "acc_adv": "停顿与重复",
        "batch_btn": "🚀 批量生成",
        "table": "结果",
        "zip": "⬇️ 下载 ZIP（音频 + 结果 CSV）",
        "footer": "文件保存在",
        "tab_files": "文件库",
        "files_hint": "输出文件夹中的文件（最新在前）— 点击一行即可播放音频或预览内容。",
        "refresh": "🔄 刷新",
        "open_folder": "📂 打开文件夹",
        "files_table": "输出文件",
        "file_player": "播放所选文件",
        "file_text": "文件内容",
        "res_cached": "♻️ 来自缓存",
        "res_new": "🆕 新生成",
        "res_file": "已保存文件",
        "loading": "正在加载模型：{model}……",
        "item_of": "第 {n} / {total} 条",
        "packaging": "正在打包结果",
        "batch_done": "**完成。**成功 {ok} 条，失败 {failed} 条，共 {total} 条。",
    },
}


def tr(lang: str, key: str) -> str:
    table = T.get(lang) or T[DEFAULT_LANG]
    return table.get(key) or T[DEFAULT_LANG].get(key, key)


def model_choices(lang: str) -> list[tuple[str, str]]:
    return [
        (tr(lang, "model_hq"), "Higher Quality — Qwen3-TTS 1.7B"),
        (tr(lang, "model_clone"), FISH_LABEL),
    ]


def mode_choices(lang: str) -> list[tuple[str, str]]:
    return [
        (tr(lang, "mode_word"), "Single Word"),
        (tr(lang, "mode_phrase"), "Vocabulary Phrase"),
        (tr(lang, "mode_sent"), "Sentence"),
        (tr(lang, "mode_para"), "Paragraph"),
    ]
