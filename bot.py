import re
import torch
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler

from eo_pl import TashkeelModel as TashkeelModelEO
from ed_pl import TashkeelModel as TashkeelModelED
from tashkeel_tokenizer import TashkeelTokenizer
from utils import remove_non_arabic

# 🔴 غير التوكن
TOKEN = "8555007260:AAGX6BPCLvoZ5feqMkmDGtjnmsfTRqY4t2U"

print("Loading models...")

# =========================
# Remove Tashkeel
# =========================
def remove_tashkeel(text):
    return re.sub(r'[\u064B-\u0652]', '', text)

# =========================
# Load Models
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = TashkeelTokenizer()

# 🔥 موديلك (EO)
eo_model = TashkeelModelEO(
    tokenizer=tokenizer,
    max_seq_len=1024,
    n_layers=6,
    learnable_pos_emb=False
)
eo_model.load_state_dict(torch.load("best_model.pt", map_location=device))
eo_model.eval().to(device)

# 🔥 موديل قوي (ED)
ed_model = TashkeelModelED(
    tokenizer=tokenizer,
    max_seq_len=1024,
    n_layers=3,
    learnable_pos_emb=False
)
ed_model.load_state_dict(torch.load("models/best_ed_mlm_ns_epoch_178.pt", map_location=device))
ed_model.eval().to(device)

print("Bot is running 🚀")

# =========================
# Mode Control
# =========================
user_mode = {}  # لكل يوزر مود

def set_eo(update, context):
    user_mode[update.effective_user.id] = "EO"
    update.message.reply_text("🧠 تم اختيار موديلك (Encoder-Only)")

def set_ed(update, context):
    user_mode[update.effective_user.id] = "ED"
    update.message.reply_text("🔥 تم اختيار الموديل القوي (Encoder-Decoder)")

def set_auto(update, context):
    user_mode[update.effective_user.id] = "AUTO"
    update.message.reply_text("⚡ تم تفعيل الوضع الذكي (أفضل نتيجة)")

# =========================
# Handle Messages
# =========================
def handle_message(update, context):
    if not update.message or not update.message.text:
        return

    text = update.message.text

    text = remove_non_arabic(text)
    text = remove_tashkeel(text)
    text = text[:500]

    if text.strip() == "":
        update.message.reply_text("❌ ابعت نص عربي بس")
        return

    mode = user_mode.get(update.effective_user.id, "AUTO")

    try:
        if mode == "EO":
            result = eo_model.do_tashkeel_batch([text], 16, False)[0]

        elif mode == "ED":
            result = ed_model.do_tashkeel_batch([text], 16, False)[0]

        else:  # AUTO
            eo_result = eo_model.do_tashkeel_batch([text], 16, False)[0]
            ed_result = ed_model.do_tashkeel_batch([text], 16, False)[0]

            # 👇 نختار الأطول (غالبًا أدق)
            result = ed_result if len(ed_result) > len(eo_result) else eo_result

        if not result:
            result = "⚠️ مش قادر أشكّل النص ده"

    except Exception as e:
        result = f"❌ Error: {str(e)}"

    update.message.reply_text(result)

# =========================
# Start Bot
# =========================
updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

# أوامر
dp.add_handler(CommandHandler("eo", set_eo))
dp.add_handler(CommandHandler("ed", set_ed))
dp.add_handler(CommandHandler("auto", set_auto))

# رسائل
dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

updater.start_polling()
updater.idle()