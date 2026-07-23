import os
import json
import difflib
import requests
from datetime import datetime

import pandas as pd
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="ایجنت هوشمند مرکز تماس", layout="wide")

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "apis_catalog.xlsx")
REQUIRED_COLUMNS = ["نام", "API", "متد", "URL", "ورودی", "خروجی", "توضیحات کامل"]


# فایل اکسل
@st.cache_data(show_spinner=False)
def load_catalog(path):
    df = pd.read_excel(path, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"ستون‌های زیر در فایل اکسل یافت نشد: {missing}")
    return df


def row_to_text(row):
    return (
        f"### {row['نام']}  (API: {row['API']})\n"
        f"- متد: {row['متد']}\n"
        f"- آدرس: {row['URL']}\n"
        f"- ورودی‌های مورد نیاز: {row['ورودی']}\n"
        f"- خروجی: {row['خروجی']}\n"
        f"- توضیحات کامل: {row['توضیحات کامل']}\n"
    )


def score_row(query, row):
    hay = " ".join([row["نام"], row["API"], row["توضیحات کامل"], row["ورودی"], row["خروجی"]])
    return difflib.SequenceMatcher(None, query, hay).ratio() * 0.4 + (
        1.0 if any(w and w in hay for w in query.split()) else 0.0
    ) * 0.6


def mock_execute(api_name):
    return {
        "status": "SUCCESS",
        "message": f"API '{api_name}' با موفقیت فراخوانی شد.",
        "called_at": datetime.now().isoformat(),
    }


#tools
def build_tools(catalog, call_log):

    @tool
    def search_api_catalog(query, top_k=3):
        """جستجوی APIهای مرتبط با درخواست کاربر."""
        scored = catalog.copy()
        scored["__score"] = scored.apply(lambda r: score_row(query, r), axis=1)
        scored = scored.sort_values("__score", ascending=False).head(top_k)
        if scored["__score"].max() <= 0.05:
            return "هیچ API مرتبطی با این درخواست پیدا نشد. از مشتری جزئیات بیشتری بپرس یا CreateComplaintTicket را برای پیگیری دستی پیشنهاد بده."
        return "\n---\n".join(row_to_text(r) for _, r in scored.iterrows())

    @tool
    def get_api_by_name(api_name):
        """جزئیات کامل یک API را دقیقاً بر اساس نام فیلد API (مثلاً GetSimPin) برمی‌گرداند."""
        match = catalog[catalog["API"].str.strip().str.lower() == api_name.strip().lower()]
        if match.empty:
            return f"هیچ APIای با نام '{api_name}' در بانک اطلاعاتی پیدا نشد."
        return row_to_text(match.iloc[0])

    @tool
    def call_api(api_name: str) -> str:
        """API شناسایی‌شده را با نام دقیق آن (فیلد API، مثلاً GetSimPin) فراخوانی/اجرا می‌کند."""
        match = catalog[catalog["API"].str.strip().str.lower() == api_name.strip().lower()]
        if match.empty:
            return json.dumps({"status": "ERROR", "message": f"API با نام '{api_name}' یافت نشد."}, ensure_ascii=False)
        row = match.iloc[0]

        result = mock_execute(row["API"])

        call_log.append(
            {
                "زمان": datetime.now().strftime("%H:%M:%S"),
                "API": row["API"],
                "نام فارسی": row["نام"],
                "متد": row["متد"],
                "آدرس": row["URL"],
                "پاسخ": result,
            }
        )
        return json.dumps(
            {
                "api": row["API"],
                "method": row["متد"],
                "url": row["URL"],
                "response": result,
            },
            ensure_ascii=False,
        )

    return [search_api_catalog, get_api_by_name, call_api]


# agent
SYSTEM_PROMPT = """تو یک ایجنت هوش مصنوعی هستی که جایگزین اپراتور باتجربه مرکز تماس همراه اول شده‌ای.
وظیفه تو دقیقاً مثل یک اپراتور باتجربه است: از بین ده‌ها/صدها فانکشنالیتی (API) موجود در سامانه،
با شنیدن نیاز مشتری، سریع و دقیق API درست را پیدا می‌کنی و همان را فراخوانی می‌کنی.

قوانین مهم:
1. هرگز API را حدسی صدا نزن. همیشه اول search_api_catalog را برای پیدا کردن API مناسب فراخوانی کن.
2. به‌محض اینکه API درست را پیدا کردی، فوراً call_api را فقط با نام همان API صدا بزن. نیازی نیست
   شماره موبایل، کدملی یا هر اطلاعات دیگری از مشتری بپرسی یا منتظر آن بمانی؛ در این نسخه صرفاً هدف
   شناسایی و فراخوانی درست API است، نه اجرای واقعی عملیات با داده‌های حساس مشتری.
3. اگر هیچ API مناسبی برای نیاز مشتری پیدا نشد، از CreateComplaintTicket برای ثبت پیگیری دستی استفاده کن.
4. پاسخ نهایی به مشتری باید کوتاه، محاوره‌ای و مودبانه باشد و فقط تایید کند که سرویس/API مربوطه
   شناسایی و فراخوانی شد (بدون اصطلاحات فنی مثل JSON یا status code).
5. همیشه به فارسی پاسخ بده.
"""


def get_agent_executor(model_name: str, catalog: pd.DataFrame, call_log: list, base_url: str):
    tools = build_tools(catalog, call_log)
    llm = ChatOpenAI(
        model=model_name,
        api_key="lm-studio",
        base_url=base_url,
        temperature=0,
    )
    agent = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    return agent


def fetch_local_models(base_url: str):
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])], None
    except Exception as e:
        return None, str(e)


# Streamlit
def main():
    st.title("ایجنت هوشمند مرکز تماس (جایگزین اپراتور)")
    st.caption("این ایجنت به‌جای اپراتور انسانی، نیاز مشتری را در بانک اطلاعاتی APIهای سامانه جست‌وجو کرده، متد درست را فراخوانی می‌کند و پاسخ می‌دهد.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "call_log" not in st.session_state:
        st.session_state.call_log = []

    try:
        catalog = load_catalog(EXCEL_PATH)
    except Exception as e:
        st.error(f"خطا در بارگذاری فایل اکسل APIها ({EXCEL_PATH}): {e}")
        st.stop()

    with st.sidebar:
        st.header("تنظیمات LM Studio")
        base_url = st.text_input("آدرس سرور", value="http://127.0.0.1:1234/v1")

        if st.button("دریافت لیست مدل‌ها"):
            models, err = fetch_local_models(base_url)
            if err:
                st.error(f"خطا در اتصال: {err}")
                st.session_state["_lmstudio_models"] = []
            else:
                st.session_state["_lmstudio_models"] = models
                if models:
                    st.success(f"{len(models)} مدل یافت شد.")
                else:
                    st.warning("هیچ مدلی در LM Studio بارگذاری نشده است.")

        available_models = st.session_state.get("_lmstudio_models", [])
        if available_models:
            model_name = st.selectbox("انتخاب مدل", available_models, index=0)
        else:
            model_name = st.text_input("شناسه مدل", value="")

    if not model_name:
        st.warning("لطفاً از نوار کناری سرور LM Studio را متصل کرده و مدل مورد نظر را انتخاب کنید.")
        st.stop()

    executor = get_agent_executor(model_name, catalog, st.session_state.call_log, base_url)

    col_chat, col_log = st.columns([2, 1])

    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input()
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            lc_messages = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    lc_messages.append(HumanMessage(content=m["content"]))
                else:
                    lc_messages.append(AIMessage(content=m["content"]))

            with st.chat_message("assistant"):
                with st.spinner("در حال جست‌وجوی API مناسب و اجرای آن..."):
                    try:
                        response = executor.invoke({"messages": lc_messages})
                        answer = response["messages"][-1].content
                    except Exception as e:
                        answer = f"خطایی رخ داد: {e}"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

    with col_log:
        st.subheader("لاگ فراخوانی APIها")
        if not st.session_state.call_log:
            st.info("هنوز هیچ APIای فراخوانی نشده است.")
        else:
            for entry in reversed(st.session_state.call_log):
                with st.expander(f"{entry['زمان']} — {entry['نام فارسی']} ({entry['API']})"):
                    st.write(f"**متد:** {entry['متد']} \n**آدرس:** `{entry['آدرس']}`")
                    st.write("**پاسخ سرویس:**")
                    st.json(entry["پاسخ"])


if __name__ == "__main__":
    main()