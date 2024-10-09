import streamlit as st
import pandas as pd
from io import StringIO
import time
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F
import io
import sys
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings('ignore')
torch.set_num_threads(16)
import cx_Oracle
import Validation as val


st.set_page_config(layout='wide',
                page_title = 'Scrubbing Correction by Machine Learning')
st.markdown("<h1 style='text-align: center;'>Welcome To Machine Learning APP for Scrubbing Team</h1>", unsafe_allow_html=True)
st.title("_Marzouk_ is :blue[cool] :sunglasses:")

password = "marzouk"

if 'password_correct' not in st.session_state:
    st.session_state['password_correct'] = False

def check_password():
    if st.session_state["password"] == password:
        st.session_state['password_correct'] = True
    else:
        st.error("Access Denied. Please enter the correct password.")

if not st.session_state['password_correct']:
    st.text_input("Enter Password", type="password", on_change=check_password, key="password")
else:
    st.success("Access Granted")
    print(val.Val_COM_ACQ_LUT('KOAX-WS-2XOR-WS','Turck'))

#Function of loading Models
    @st.cache_resource(show_spinner="Loading Model, Just 10 Seconds")
    def load_models():
        # Load the model and tokenizer KOA
        model_koa = T5ForConditionalGeneration.from_pretrained("./corrector_model_KOA")
        tokenizer_koa = T5Tokenizer.from_pretrained("./corrector_model_KOA")
        # Load the model and tokenizer Multi Suppliers ()
        model_multi = T5ForConditionalGeneration.from_pretrained("./corrector_model_Multi")
        tokenizer_multi = T5Tokenizer.from_pretrained("./corrector_model_Multi")
        # Load the model and tokenizer Vishay alias Supplier
        model_alias = T5ForConditionalGeneration.from_pretrained("./corrector_model_Alias_Vishay")
        tokenizer_alias = T5Tokenizer.from_pretrained("./corrector_model_Alias_Vishay")
        return model_koa,tokenizer_koa,model_multi,tokenizer_multi,model_alias,tokenizer_alias

    # loading Models
    model_koa,tokenizer_koa,model_multi,tokenizer_multi,model_alias,tokenizer_alias = load_models()


    # OLD Way for Validation on Component
    _='''
    def Get_Validation(corrected_Part,Se_Supplier):
        # Database connection parameters
        dsn = cx_Oracle.makedsn("10.199.104.127", "1521", service_name="scrubbing")
        connection = cx_Oracle.connect(user="Read_Only", password="READ_ONLY", dsn=dsn)
        cursor = connection.cursor()
        # Create a string of MPNs for SQL IN clause
        input_str =', '.join([f"'{corrected_Part}'"])
        input_supplier= ', '.join([f"'{Se_Supplier}'"])
        #corrected_Part=str(corrected_Part)
        #Se_Supplier=str(Se_Supplier)
        # SQL query
        query = f"SELECT com_id, com_partnum , cm.get_man_name(man_id) as se_supplier FROM cm.xlp_se_component WHERE NAN_PARTNUM = NONALPHA({input_str}) and cm.get_man_name(man_id)={input_supplier}"
        cursor.execute(query)
        rows=cursor.fetchall()
        dict_r={}
        lst_r=[]
        if rows:
            for row in rows:
                dict_r['COM_ID']=row[0]
                dict_r['COM_PARTNUM']=row[1]
                dict_r['SE_SUPPLIER']=row[2]
                lst_r.append(dict_r)
                return pd.DataFrame(lst_r)
        else:
            return pd.DataFrame([{'COM_ID':'','COM_PARTNUM':'','SE_SUPPLIER':''}])
        cursor.close()
        connection.close()
'''

    # Old Way for Validation on Component for List Files
    _='''
    def Get_Validation_for_List_File(corrected_Part,Se_Supplier):
        # Database connection parameters
        dsn = cx_Oracle.makedsn("10.199.104.127", "1521", service_name="scrubbing")
        connection = cx_Oracle.connect(user="Read_Only", password="READ_ONLY", dsn=dsn)
        cursor = connection.cursor()
        # Create a string of MPNs for SQL IN clause
        input_str =', '.join([f"'{corrected_Part}'"])
        input_supplier= ', '.join([f"'{Se_Supplier}'"])
        # SQL query
        query = f"SELECT com_id, com_partnum , cm.get_man_name(man_id) as se_supplier FROM cm.xlp_se_component WHERE NAN_PARTNUM = NONALPHA({input_str}) and cm.get_man_name(man_id)={input_supplier}"
        cursor.execute(query)
        rows=cursor.fetchall()
        if rows:
            for row in rows:
                Val_Type='Exact'
                COM_ID=row[0]
                COM_PARTNUM=row[1]
                SE_SUPPLIER=row[2]
                break
            return Val_Type,COM_ID,COM_PARTNUM,SE_SUPPLIER
        else:
            return 'Not_Match','','',''
        cursor.close()
        connection.close()
'''



    # Function to calculate confidence score
    def calculate_confidence(logits, predicted_tokens):
        probabilities = [F.softmax(logit, dim=-1) for logit in logits]
        predicted_probs = [prob[0, token].item() for prob, token in zip(probabilities, predicted_tokens[0])]
        confidence = sum(predicted_probs) / len(predicted_probs)
        return confidence

        # Example inference with confidence score

    def correct_Koa_MPN_with_confidence(input_text):
        input_text=str(input_text)
        input_ids = tokenizer_koa("correct: " + input_text, return_tensors="pt").input_ids
        outputs = model_koa.generate(input_ids, output_scores=True, return_dict_in_generate=True,max_length=200)
        # Decode the generated tokens to get the corrected text
        corrected_MPN = tokenizer_koa.decode(outputs.sequences[0], skip_special_tokens=True)
        # Calculate the confidence score
        confidence = calculate_confidence(outputs.scores, outputs.sequences[:, 1:])
        return corrected_MPN, confidence

    def correct_MPN_Multi_with_confidence(input_text, supplier_name):
        input_text=str(input_text)
        # Combine input text with supplier name
        combined_input = f"correct: {input_text} supplier: {supplier_name}"
        # Tokenize the combined input
        input_ids = tokenizer_multi(combined_input, return_tensors="pt").input_ids
        # Generate output with scores
        outputs = model_multi.generate(input_ids, output_scores=True, return_dict_in_generate=True)
        # Decode the generated tokens to get the corrected text
        corrected_MPN = tokenizer_multi.decode(outputs.sequences[0], skip_special_tokens=True)
        # Calculate the confidence score
        confidence = calculate_confidence(outputs.scores, outputs.sequences[:, 1:])
        return corrected_MPN, confidence

    def correct_Vishay_Alias_MPN_with_confidence(input_text):
        input_text=str(input_text)
        input_ids = tokenizer_alias("correct: " + input_text, return_tensors="pt").input_ids
        outputs = model_alias.generate(input_ids, output_scores=True, return_dict_in_generate=True,max_length=200)
        # Decode the generated tokens to get the corrected text
        corrected_MPN = tokenizer_alias.decode(outputs.sequences[0], skip_special_tokens=True)
        # Calculate the confidence score
        confidence = calculate_confidence(outputs.scores, outputs.sequences[:, 1:])
        return corrected_MPN, confidence



    def clear_session_state():
        for key in st.session_state.keys():
            del st.session_state[key]

    # loading all data frame to use it in dashboard


    def main():
        
        
        if 'page' not in st.session_state:
            st.session_state.page = 'Correction APP'
            
        # Side bar with box selection
        box = st.sidebar.radio(label='Type', options=['Correction APP','Alias APP'],key =1,label_visibility = "visible")
        st.session_state.page = box
        if st.session_state.page == 'Correction APP':
            
            html_temp = """
            <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;">Scrubbing Correction by Machine Learning APP </h2>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)
            
        
            SUPPLIER_NAME = st.selectbox("Supplier Name",['TT Electronics','Murata Manufacturing','Yageo','KYOCERA AVX Components Corporation','Stackpole Electronics, Inc','KOA Speer Electronics','Panasonic','onsemi','Texas Instruments','ROHM Semiconductor','Analog Devices','KEMET Corporation','TDK','Microchip Technology','TE Connectivity','Infineon Technologies AG','Molex','Renesas Electronics','NXP Semiconductors'])
            
            
            input_MPN=st.text_input(label="Insert MPN to Correct",placeholder='Marzouk',value='Marzouk')
            result=""
            accuracy=0
            
            
            if st.button("Correct"):
                
                if SUPPLIER_NAME == 'KOA Speer Electronics':
                    if input_MPN !='Marzouk' and  input_MPN:
                        result,accuracy  = correct_Koa_MPN_with_confidence(input_MPN)
                        accuracy_percent = f"{accuracy:.2%}"
                        print('The Corrected Part is: {} | with Accuracy {} %'.format(result,accuracy_percent),flush=True)
                        st.success('The Corrected Part is: {}'.format(result),icon="✅")
                        st.success('Accuracy: {}'.format(accuracy_percent),icon="✅")
                        if accuracy>=0.8:
                            st.dataframe(Get_Validation(result,SUPPLIER_NAME),hide_index=True)

                    else:
                        st.warning('Wrong input, Please try another inputs', icon="⚠️")
                elif SUPPLIER_NAME in ['Murata Manufacturing','Yageo','KYOCERA AVX Components Corporation','Stackpole Electronics, Inc','TT Electronics','Panasonic','onsemi','Texas Instruments','ROHM Semiconductor','Analog Devices','KEMET Corporation','TDK','Microchip Technology','TE Connectivity','Infineon Technologies AG','Molex','Renesas Electronics','NXP Semiconductors']:
                    if input_MPN !='Marzouk' and  input_MPN:
                        result,accuracy  = correct_MPN_Multi_with_confidence(input_MPN,SUPPLIER_NAME)
                        accuracy_percent = f"{accuracy:.2%}"
                        print('The Corrected Part is: {} | with Accuracy {} %'.format(result,accuracy_percent),flush=True)
                        st.success('The Corrected Part is: {}'.format(result),icon="✅")
                        st.success('Accuracy: {}'.format(accuracy_percent),icon="✅")
                        if accuracy>=0.8:
                            st.dataframe(Get_Validation(result,SUPPLIER_NAME),hide_index=True)
                    else:
                        st.warning('Wrong input, Please try another inputs', icon="⚠️")
                else:
                    st.warning('Supplier Still under Training Phase', icon="⚠️")
                        
                    
            # Upload File to predecit list of Rows
            
            #st.write("Please upload a text file with the following headers: SUPPLIER_NAME, FUNCTION_CATEGORY, PL, PoC, RN")
            # File uploader
            uploaded_file = st.file_uploader("Choose a file on header [MPN,Supplier]", type=['xlsx'])
            if uploaded_file is not None:
                try:
                # To read file afrom excel:
                    df_input = pd.read_excel(uploaded_file, engine='openpyxl')

                    # Check if all required columns are present
                    expected_columns = ['MPN','Supplier']
                    if all(column in df_input.columns for column in expected_columns):
                        # Display the dataframe
                        st.write(df_input)
                    else:
                        st.error("Uploaded file does not contain the required header, [MPN]")
                except Exception as e:
                    st.error(f"Error reading the file: {e}")
            if uploaded_file is not None:
                df_input = pd.read_excel(uploaded_file, engine='openpyxl')
                MPNlist=df_input['MPN'].to_list()
                Suppliers=df_input['Supplier'].to_list()
                result_dict={}
                st.session_state.MPN_LIST = []
                st.session_state.Supplier_List = []
                st.session_state.corrected_part_list = []
                st.session_state.accuracy_list = []
                st.session_state.Flag=[]
                st.session_state.Val_Type=[]
                st.session_state.COM_ID=[]
                st.session_state.COM_PARTNUM=[]
                st.session_state.SE_SUPPLIER=[]
                
                df_result=pd.DataFrame(result_dict)
                csv=df_result.to_csv(index=False)
                if st.button("Correct List of MPN"):
                    # Initialize progress bar
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    total_rows = len(MPNlist)
                    #with st.spinner('Processing...'):
                    for idx, (MPN, Supplier) in enumerate(zip(MPNlist, Suppliers)):
                        if Supplier in ['Murata Manufacturing','Yageo','KYOCERA AVX Components Corporation','Stackpole Electronics, Inc','TT Electronics','Panasonic','onsemi','Texas Instruments','ROHM Semiconductor','Analog Devices','KEMET Corporation','TDK','Microchip Technology','TE Connectivity','Infineon Technologies AG','Molex','Renesas Electronics','NXP Semiconductors']:
                            corrected_part, accuracy = correct_MPN_Multi_with_confidence(MPN,Supplier)
                            accuracy = f"{accuracy:.2%}"
                            #print(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}",flush=True)
                            #st.write(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}",unsafe_allow_html=True)
                            #st.markdown(f"**{MPN}** | **{Supplier}** | **{corrected_part}** | **{accuracy}**")
                            
                            st.session_state.MPN_LIST.append(MPN)
                            st.session_state.Supplier_List.append(Supplier)
                            st.session_state.corrected_part_list.append(corrected_part)
                            st.session_state.accuracy_list.append(accuracy)
                            st.session_state.Flag.append(1)
                            Val_Type,COM_ID,COM_PARTNUM,SE_SUPPLIER=Get_Validation_for_List_File(corrected_part,Supplier)
                            st.session_state.Val_Type.append(Val_Type)
                            st.session_state.COM_ID.append(COM_ID)
                            st.session_state.COM_PARTNUM.append(COM_PARTNUM)
                            st.session_state.SE_SUPPLIER.append(SE_SUPPLIER)
                            print(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}\t{Val_Type}\t{COM_ID}\t{COM_PARTNUM}\t{SE_SUPPLIER}",flush=True)
                        elif Supplier=='KOA Speer Electronics':
                            corrected_part, accuracy = correct_Koa_MPN_with_confidence(MPN)
                            accuracy = f"{accuracy:.2%}"
                            #print(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}",flush=True)
                            #st.write(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}",unsafe_allow_html=True)
                            #st.markdown(f"**{MPN}** | **{Supplier}** | **{corrected_part}** | **{accuracy}**")
                            
                            st.session_state.MPN_LIST.append(MPN)
                            st.session_state.Supplier_List.append(Supplier)
                            st.session_state.corrected_part_list.append(corrected_part)
                            st.session_state.accuracy_list.append(accuracy)
                            st.session_state.Flag.append(1)
                            Val_Type,COM_ID,COM_PARTNUM,SE_SUPPLIER=Get_Validation_for_List_File(corrected_part,Supplier)
                            st.session_state.Val_Type.append(Val_Type)
                            st.session_state.COM_ID.append(COM_ID)
                            st.session_state.COM_PARTNUM.append(COM_PARTNUM)
                            st.session_state.SE_SUPPLIER.append(SE_SUPPLIER)
                            print(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}\t{Val_Type}\t{COM_ID}\t{COM_PARTNUM}\t{SE_SUPPLIER}",flush=True)
                        else:
                            corrected_part, accuracy = correct_MPN_Multi_with_confidence(MPN,"")
                            accuracy = f"{accuracy:.2%}"
                            #print(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}\tGeneral Correction",flush=True)
                            #st.write(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}",unsafe_allow_html=True)
                            #st.markdown(f"**{MPN}** | **{Supplier}** | **{corrected_part}** | **{accuracy}**")
                            
                            st.session_state.MPN_LIST.append(MPN)
                            st.session_state.Supplier_List.append(Supplier)
                            st.session_state.corrected_part_list.append(corrected_part)
                            st.session_state.accuracy_list.append(accuracy)
                            st.session_state.Flag.append(0)
                            Val_Type,COM_ID,COM_PARTNUM,SE_SUPPLIER=Get_Validation_for_List_File(corrected_part,Supplier)
                            st.session_state.Val_Type.append(Val_Type)
                            st.session_state.COM_ID.append(COM_ID)
                            st.session_state.COM_PARTNUM.append(COM_PARTNUM)
                            st.session_state.SE_SUPPLIER.append(SE_SUPPLIER)
                            print(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}\t{Val_Type}\t{COM_ID}\t{COM_PARTNUM}\t{SE_SUPPLIER}",flush=True)
                        
                        # Update progress bar
                        progress = (idx + 1) / total_rows
                        progress_bar.progress(progress)
                        progress_text.text(f"Processing row {idx + 1} of {total_rows}")
                            
                    # Return Finished Parts as dataframe to can download it    
                result_dict['MPN']=st.session_state.MPN_LIST
                result_dict['Suppliers']=st.session_state.Supplier_List
                result_dict['corrected_part']=st.session_state.corrected_part_list
                result_dict['accuracy']=st.session_state.accuracy_list
                result_dict['Flag_Trained']=st.session_state.Flag
                result_dict['Val_Type']=st.session_state.Val_Type
                result_dict['COM_ID']=st.session_state.COM_ID
                result_dict['COM_PARTNUM']=st.session_state.COM_PARTNUM
                result_dict['SE_SUPPLIER']=st.session_state.SE_SUPPLIER
                df_result=pd.DataFrame(result_dict)
                csv=df_result.to_csv(index=False)
                
                # Create a download button
                st.download_button(
                    label="Download Result",
                    data=csv,
                    file_name='corrected_data.csv',
                    mime='text/csv',
                    disabled=False
                )
                        #clear_session_state()
                            
            else:
                st.button("Correct List of MPN",disabled=True)
                
    ## Alias Page
                
        elif st.session_state.page  == 'Alias APP':
            
            html_temp = """
            <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;">Alias by Machine Learning APP </h2>
            </div>
                """
            st.markdown(html_temp,unsafe_allow_html=True)
            
            SUPPLIER_NAME_1 = st.selectbox("Supplier Name",['Vishay'])
            
            
            input_MPN_1=st.text_input(label="Insert MPN to Correct",placeholder='Marzouk',value='Marzouk')
            result=""
            accuracy=0
            
            
            if st.button("Correct"):
                if SUPPLIER_NAME_1 == 'Vishay':
                    if input_MPN_1 !='Marzouk' and  input_MPN_1:
                        result,accuracy  = correct_Vishay_Alias_MPN_with_confidence(input_MPN_1)
                        accuracy_percent = f"{accuracy:.2%}"
                        print('The Corrected Part is: {} | with Accuracy {} %'.format(result,accuracy_percent),flush=True)
                        st.success('The Corrected Part is: {}'.format(result),icon="✅")
                        st.success('Accuracy: {}'.format(accuracy_percent),icon="✅")
                        if accuracy>=0.8:
                            st.dataframe(Get_Validation(result,SUPPLIER_NAME_1),hide_index=True)
                    else:
                        st.warning('Wrong input, Please try another inputs', icon="⚠️")
                else:
                    st.warning('Supplier Still under Training Phase', icon="⚠️")
                        
                    
            # Upload File to predecit list of Rows
            
            #st.write("Please upload a text file with the following headers: SUPPLIER_NAME, FUNCTION_CATEGORY, PL, PoC, RN")
            # File uploader
            uploaded_file = st.file_uploader("Choose a file on header [MPN,Supplier]", type=['xlsx'])
            if uploaded_file is not None:
                try:
                # To read file afrom excel:
                    df_input = pd.read_excel(uploaded_file, engine='openpyxl')

                    # Check if all required columns are present
                    expected_columns = ['MPN','Supplier']
                    if all(column in df_input.columns for column in expected_columns):
                        # Display the dataframe
                        st.write(df_input)
                    else:
                        st.error("Uploaded file does not contain the required header, [MPN]")
                except Exception as e:
                    st.error(f"Error reading the file: {e}")
            if uploaded_file is not None:
                
                df_input = pd.read_excel(uploaded_file, engine='openpyxl')
                MPNlist=df_input['MPN'].to_list()
                Suppliers=df_input['Supplier'].to_list()
                result_dict={}
                MPN_LIST=[]
                Supplier_List=[]
                corrected_part_list=[]
                accuracy_list=[]
                
                st.session_state.MPN_LIST = []
                st.session_state.Supplier_List = []
                st.session_state.corrected_part_list = []
                st.session_state.accuracy_list = []
                st.session_state.Flag=[]
                st.session_state.Val_Type=[]
                st.session_state.COM_ID=[]
                st.session_state.COM_PARTNUM=[]
                st.session_state.SE_SUPPLIER=[]
                
                
                if st.button("Get New Format of Old MPNs"):
                    
                    # Initialize progress bar
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    total_rows = len(MPNlist)
                    #with st.spinner('Processing...'):
                    for idx, (MPN, Supplier) in enumerate(zip(MPNlist, Suppliers)):
                        if Supplier=='Vishay':
                            corrected_part, accuracy = correct_Vishay_Alias_MPN_with_confidence(MPN)
                            accuracy = f"{accuracy:.2%}"
                            #print(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}",flush=True)
                            #st.write(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}",unsafe_allow_html=True)
                            #st.markdown(f"**{MPN}** | **{Supplier}** | **{corrected_part}** | **{accuracy}**")
                            
                            st.session_state.MPN_LIST.append(MPN)
                            st.session_state.Supplier_List.append(Supplier)
                            st.session_state.corrected_part_list.append(corrected_part)
                            st.session_state.accuracy_list.append(accuracy)
                            st.session_state.Flag.append(1)
                            Val_Type,COM_ID,COM_PARTNUM,SE_SUPPLIER=Get_Validation_for_List_File(corrected_part,Supplier)
                            st.session_state.Val_Type.append(Val_Type)
                            st.session_state.COM_ID.append(COM_ID)
                            st.session_state.COM_PARTNUM.append(COM_PARTNUM)
                            st.session_state.SE_SUPPLIER.append(SE_SUPPLIER)
                            print(f"{MPN}\t{Supplier}\t{corrected_part}\t{accuracy}\t{Val_Type}\t{COM_ID}\t{COM_PARTNUM}\t{SE_SUPPLIER}",flush=True)
                        else:
                            st.warning('there are Supplier into list that Still under Training Phase', icon="⚠️")
                            break
                        
                        # Update progress bar
                        progress = (idx + 1) / total_rows
                        progress_bar.progress(progress)
                        progress_text.text(f"Processing row {idx + 1} of {total_rows}")
                        
                        
                        _="""
                        if stop_process:
                            result_dict['MPN']=st.session_state.MPN_LIST
                            result_dict['Suppliers']=st.session_state.Supplier_List
                            result_dict['corrected_part']=st.session_state.corrected_part_list
                            result_dict['accuracy']=st.session_state.accuracy_list
                            result_dict['Flag']=st.session_state.Flag
                            df_result=pd.DataFrame(result_dict)
                            csv=df_result.to_csv(index=False)
                            
                            # Create a download button
                            st.download_button(
                                label="Download Result",
                                data=csv,
                                file_name='corrected_data.csv',
                                mime='text/csv'
                            )
                            break
                            """
                    # Return Finished Parts as dataframe to can download it    
                    result_dict['MPN']=st.session_state.MPN_LIST
                    result_dict['Suppliers']=st.session_state.Supplier_List
                    result_dict['corrected_part']=st.session_state.corrected_part_list
                    result_dict['accuracy']=st.session_state.accuracy_list
                    result_dict['Flag_Trained']=st.session_state.Flag
                    result_dict['Val_Type']=st.session_state.Val_Type
                    result_dict['COM_ID']=st.session_state.COM_ID
                    result_dict['COM_PARTNUM']=st.session_state.COM_PARTNUM
                    result_dict['SE_SUPPLIER']=st.session_state.SE_SUPPLIER
                    df_result=pd.DataFrame(result_dict)
                    csv=df_result.to_csv(index=False)
                    
                    # Create a download button
                    st.download_button(
                        label="Download Result",
                        data=csv,
                        file_name='corrected_data.csv',
                        mime='text/csv'
                    )
                        #clear_session_state()
                        
                        
            else:
                st.button("Correct List of MPN",disabled=True)
            
            
    if __name__=='__main__':
        main()