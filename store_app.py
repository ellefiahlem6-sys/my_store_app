import streamlit as st
import pandas as pd
import os
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="متجري الإلكتروني", layout="wide", page_icon="🛒")

# إضافة بعض التنسيق البسيط للغة العربية (اتجاه من اليمين لليسار)
st.markdown("""
<style>
    .rtl { direction: rtl; text-align: right; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("🛒 نظام إدارة المتجر الإلكتروني")
st.markdown("---")

# 2. إعداد قاعدة البيانات (ملفات CSV)
if not os.path.exists('products.csv'):
    pd.DataFrame(columns=['ID', 'Name', 'Price', 'Stock']).to_csv('products.csv', index=False)
if not os.path.exists('sales.csv'):
    pd.DataFrame(columns=['Date', 'Product_ID', 'Product_Name', 'Quantity', 'Total_Price']).to_csv('sales.csv', index=False)

# 3. القائمة الجانبية للتنقل
st.sidebar.title("القائمة الرئيسية")
menu = st.sidebar.radio("اذهب إلى:", ["🏠 لوحة التحكم", "🤖 التنبؤ بالمبيعات",  "📦 إدارة المخزون", "💰 المبيعات", "📞 تواصل معنا"])

# ==========================================
# 4. صفحة لوحة التحكم (Dashboard)
# ==========================================
if menu == "🏠 لوحة التحكم":
    st.header("📊 نظرة عامة على المتجر")
    
    df_products = pd.read_csv('products.csv')
    df_sales = pd.read_csv('sales.csv')
    
    # حساب الإحصائيات
    total_revenue = df_sales['Total_Price'].sum() if not df_sales.empty else 0
    total_products = len(df_products)
    total_sales_count = len(df_sales)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("إجمالي الإيرادات", f"{total_revenue:,.2f} دينار")
    col2.metric("عدد المنتجات", total_products)
    col3.metric("عدد عمليات البيع", total_sales_count)
    
    st.markdown("---")
    st.subheader("⚠️ تنبيهات المخزون (المنتجات التي أوشكت على النفاد)")
    low_stock = df_products[df_products['Stock'] < 5]
    if not low_stock.empty:
        st.warning("هذه المنتجات تحتاج إلى إعادة تخزين:")
        st.dataframe(low_stock, use_container_width=True, hide_index=True)
    else:
        st.success("المخزون بحالة جيدة! ✅")

# ==========================================
# 5. صفحة إدارة المخزون (Inventory)
# ==========================================
elif menu == "📦 إدارة المخزون":
    st.header("📦 إدارة المنتجات")
    
    df_products = pd.read_csv('products.csv')
    
    # عرض المخزون الحالي
    st.subheader("المنتجات الحالية")
    if not df_products.empty:
        st.dataframe(df_products, use_container_width=True, hide_index=True)
    else:
        st.info("لا توجد منتجات حالياً. أضف منتجك الأول!")
        
    st.markdown("---")
    
    # إضافة منتج جديد
    st.subheader("➕ إضافة منتج جديد")
    with st.form("add_product_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            prod_id = st.text_input("كود المنتج (ID)")
        with col2:
            prod_name = st.text_input("اسم المنتج")
        with col3:
            prod_price = st.number_input("السعر (دينار)", min_value=0.0, format="%.2f")
            
        prod_stock = st.number_input("الكمية المتاحة", min_value=0, step=1)
        
        submit_add = st.form_submit_button("إضافة المنتج")
        
        if submit_add:
            if prod_id and prod_name and prod_price > 0:
                # التحقق من عدم تكرار الكود
                if prod_id in df_products['ID'].values:
                    st.error("هذا الكود مستخدم بالفعل! اختر كوداً آخر.")
                else:
                    new_product = pd.DataFrame({'ID': [prod_id], 'Name': [prod_name], 'Price': [prod_price], 'Stock': [prod_stock]})
                    new_product.to_csv('products.csv', mode='a', header=False, index=False)
                    st.success(f"تم إضافة المنتج {prod_name} بنجاح! 🎉")
                    st.rerun() # تحديث الصفحة لرؤية المنتج الجديد
            else:
                st.error("الرجاء تعبئة جميع الحقول بشكل صحيح.")

# ==========================================
# 6. صفحة المبيعات (Sales)
# ==========================================
elif menu == "💰 المبيعات":
    st.header("💰 تسجيل عملية بيع")
    
    df_products = pd.read_csv('products.csv')
    
    if df_products.empty:
        st.warning("لا توجد منتجات في المخزون. أضف منتجات أولاً من صفحة إدارة المخزون.")
    else:
        # اختيار المنتج
        product_options = df_products.set_index('ID')['Name'].to_dict()
        selected_id = st.selectbox("اختر المنتج:", options=list(product_options.keys()), format_func=lambda x: f"{x} - {product_options[x]}")
        
        # الحصول على بيانات المنتج المختار
        product_data = df_products[df_products['ID'] == selected_id].iloc[0]
        max_stock = product_data['Stock']
        price = product_data['Price']
        
        st.info(f"السعر: {price} دينار | الكمية المتاحة: {max_stock}")
        
        # إدخال الكمية
        qty = st.number_input("الكمية المباعة", min_value=1, max_value=max_stock, step=1)
        total = qty * price
        st.write(f"### الإجمالي: {total:,.2f} دينار")
        
        if st.button("تأكيد عملية البيع", type="primary"):
            if max_stock >= qty:
                # 1. تحديث المخزون
                df_products.loc[df_products['ID'] == selected_id, 'Stock'] -= qty
                df_products.to_csv('products.csv', index=False)
                
                # 2. تسجيل عملية البيع
                new_sale = pd.DataFrame({
                    'Date': [datetime.now().strftime("%Y-%m-%d %H:%M")],
                    'Product_ID': [selected_id],
                    'Product_Name': [product_data['Name']],
                    'Quantity': [qty],
                    'Total_Price': [total]
                })
                new_sale.to_csv('sales.csv', mode='a', header=False, index=False)
                
                st.success(f"تمت عملية البيع بنجاح! تم بيع {qty} من {product_data['Name']}.")
                st.balloons()
                st.rerun()
            else:
                st.error("الكمية المطلوبة غير متوفرة في المخزون!")

# ==========================================
# 7. صفحة تواصل معنا (Contact)
# ==========================================
elif menu == "📞 تواصل معنا":
    st.header("📞 خدمة العملاء")
    st.write("هل لديك أي استفسار أو شكوى؟ اترك رسالتك وسنقوم بالرد عليك في أقرب وقت.")
    
    with st.form("contact_form"):
        name = st.text_input("الاسم الكامل")
        email = st.text_input("البريد الإلكتروني")
        message = st.text_area("رسالتك")
        
        submit_contact = st.form_submit_button("إرسال الرسالة")
        
        if submit_contact:
            if name and email and message:
                st.success("تم إرسال رسالتك بنجاح! شكراً لتواصلك معنا. ✅")
                # في تطبيق حقيقي، هنا سنقوم بإرسال بريد إلكتروني أو حفظ الرسالة في قاعدة بيانات
            else:
                st.error("الرجاء تعبئة جميع الحقول.")

# ==========================================
# 8. صفحة التنبؤ بالمبيعات (ML Prediction)
# ==========================================
elif menu == "🤖 التنبؤ بالمبيعات":
    st.header("🤖 التنبؤ الذكي بمبيعات الغد")
    st.write("هذا القسم يستخدم تعلم الآلة (Random Forest) للتنبؤ بإيرادات الغد بناءً على بيانات مبيعاتك السابقة.")
    
    # 1. قراءة بيانات المبيعات
    try:
        df_sales = pd.read_csv('sales.csv')
    except FileNotFoundError:
        df_sales = pd.DataFrame()
        
    if df_sales.empty or len(df_sales) < 5:
        st.warning("⚠️ لا توجد بيانات كافية للتدريب. قم بتسجيل بعض المبيعات أولاً من صفحة (المبيعات).")
        
        # --- توليد بيانات وهمية للتوضيح فقط ---
        st.info("💡 لأغراض العرض، سنستخدم بيانات وهمية لتوضيح كيف يعمل الموديل.")
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        dummy_sales = pd.DataFrame({
            'Date': dates,
            'Total_Price': np.random.randint(500, 3000, size=60) # مبيعات عشوائية بين 500 و 3000
        })
        df_sales = dummy_sales
    else:
        # تحويل عمود التاريخ إلى صيغة تاريخ حقيقية
        df_sales['Date'] = pd.to_datetime(df_sales['Date'])
        # تجميع المبيعات حسب اليوم (لأن المستخدم قد يبيع عدة منتجات في نفس اليوم)
        df_sales = df_sales.groupby('Date')['Total_Price'].sum().reset_index()

    # 2. هندسة الخصائص (Feature Engineering)
    # نستخرج من التاريخ: يوم الأسبوع، اليوم في الشهر، الشهر
    df_sales['DayOfWeek'] = df_sales['Date'].dt.dayofweek
    df_sales['Day'] = df_sales['Date'].dt.day
    df_sales['Month'] = df_sales['Date'].dt.month

    X = df_sales[['DayOfWeek', 'Day', 'Month']]
    y = df_sales['Total_Price']

    # 3. تدريب الموديل
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_sales = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model_sales.fit(X_train, y_train)
    
    # 4. تقييم الموديل
    score = model_sales.score(X_test, y_test)
    st.metric("دقة الموديل (R² Score)", f"{score:.2f}")

    # 5. التنبؤ لمبيعات الغد
    if st.button("توقع مبيعات الغد 🚀", type="primary"):
        tomorrow = datetime.now() + pd.Timedelta(days=1)
        tomorrow_features = pd.DataFrame({
            'DayOfWeek': [tomorrow.dayofweek],
            'Day': [tomorrow.day],
            'Month': [tomorrow.month]
        })
        
        predicted_revenue = model_sales.predict(tomorrow_features)[0]
        
        st.success(f"🔮 الإيرادات المتوقعة للغد: **{predicted_revenue:,.2f} دينار**")
        st.balloons()