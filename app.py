import streamlit as st
import pickle
import torch
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Hugging Face LLM models Experimentation",
    page_icon="😊",
    layout="wide"
)

# Load the best performing model and tokenizer
@st.cache_resource
def load_model():
    with open(r'C:\Users\11 PrO\Desktop\trying bots\dist\model_3_dist.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(r'C:\Users\11 PrO\Desktop\trying bots\dist\tokenizer_3_dist.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Function to predict emotion
def predict_emotion(text, model, tokenizer):
    # Preprocess and tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_label].item()
    
    # Map predictions to emotions
    emotion_map = {0: "Joy", 1: "Sadness", 2: "Anger", 3: "Fear"}
    return emotion_map[predicted_label], confidence

# Main app
def main():
    st.title("🤗 Hugging Face LLM models Experimentation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dataset Information", "Model 1", "Model 2", "Model 3"]
    )
    
    # Dataset Information Page
    if page == "Dataset Information":
        st.header("Dataset Information")
        st.write("""
        ### About the Dataset
        The emotion dataset used for training these models consists of text samples labeled with different emotions. 
        The dataset was modified to contain 1000 records and the same dataset has been used to experiment with the different models.
        It comprises of two columns: 
        -text:- dialog
        -label:-emotion 

        The dialogs in the dataset were preprocessed and mapped to four primary emotions:
        
        - Joy 😊-1
        - Sadness 😢-0
        - Anger 😠-3
        - Fear 😨-4
        """)
        
        st.markdown("---")  # Page break
        st.header("Data Distribution")
        try:
            dataset_dist = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\dataset.png')
            st.image(dataset_dist, caption="Dataset Used")
        except:
            st.write("Dataset distribution visualization not available")
    
    # Model 1: micro Analysis Page
    elif page == "Model 1":
        st.header("microsoft/deberta-base Model Analysis")
        
        st.subheader("Model Hyperparameter Analysis")
        try:
            micro_x = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\micro_x.png')
            st.image(micro_x, caption="Findings")
            st.markdown("---")
            micro_metrics = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\micro\analysis_micro.png')
            st.image(micro_metrics, caption="Model Hyperparameter Analysis")
        except:
            st.write("micro metrics visualization not available")
        
        st.markdown("---")  # Page break
        
        st.subheader("Training Performance")
        try:
            micro_test_graph = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\micro\training_metrics_micro_1.png')
            st.image(micro_test_graph, caption="Accuracy and Loss Curves")
        except:
            st.write("Training metrics not available")
            
        st.markdown("---")  # Page break
        
        st.subheader("Test Results")
        try:
            micro_test_results = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\micro\test1micro.png')
            st.image(micro_test_results, caption="Best Performing Model's test Results")
            st.write("The model has an accuracy of 72%")
        except:
            st.write("Test results not available")
        
        st.markdown("---")  # Page break
        
        st.write("""
        ### Model Performance Summary
        - Architecture: microsoft/deberta-base
        - Fine-tuned on emotion dataset
        - Best validation accuracy achieved so far-72%
        - Need to provide a lower Batch Size since the model is huge and complex(restricted to 4 in this case)
        """)
    
    # Model 2: cadiff Analysis Page
    elif page == "Model 2":
        st.header("cardiffnlp/twitter-roberta-base-emotion Model Analysis")
        
        st.subheader("Model Hyperparameter Analysis")
        try:
            cadiff_x= Image.open(r'C:\Users\11 PrO\Desktop\trying bots\cadiff_x.png')
            st.image(cadiff_x, caption="findings")
            st.markdown("---")  
            cadiff_metrics = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\cadiff_analysis.png')
            st.image(cadiff_metrics, caption="cardiffnlp/twitter-roberta-base-emotion hyperparameter analysis")
        except:
            st.write("cadiff metrics visualization not available")
        
        st.markdown("---")  # Page break
        
        st.subheader("Training Performance")
        try:
            cadiff_test_graph = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\training_metrics_cadiff_3.png')
            st.image(cadiff_test_graph, caption="Accuracy and Loss Curves")
        except:
            st.write("Training metrics not available")
            
        st.markdown("---")  # Page break
        
        st.subheader("Test Results")
        try:
            cadiff_test_results = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\cadiff_test_3.png')
            st.image(cadiff_test_results, caption="Best Performing Model's test Results")
            st.write("The model has an accuracy of 67%")
        except:
            st.write("Test results not available")
        
        st.markdown("---")  # Page break
        
        st.write("""
        ### Model Performance Summary
        - Architecture: cardiffnlp/twitter-roberta-base-emotion
        - Fine-tuned on emotion dataset
        - Best validation accuracy achieved so far-67%
        """)
        st.write("""
                 ### Drawback:
                 -The model was not able to differentiate between the emotions sadness and anger
                 """)
    
    # Model 3: Distilcadiff Analysis Page
    elif page == "Model 3":
        st.header("Distilbert Model Analysis")
        
        st.subheader("Model Hyperparameter Analysis")
        try:
            distil_x= Image.open(r'C:\Users\11 PrO\Desktop\trying bots\dist_x.png')
            st.image(distil_x, caption="findings")
            st.markdown("---")
            st.write("Weight Decay Analysis")
            distilcadiff_weight = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\weight_accuracy.png')
            st.image(distilcadiff_weight, caption="Weight Accuracy Analysis")
            
            st.markdown("---")  # Page break
            
            st.write("Learning Rate Analysis")
            distilcadiff_learning = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\dist\learning_accuracy.png')
            st.image(distilcadiff_learning, caption="Accuracy Learning Rate Analysis")
            
            st.markdown("---")  # Page break
            
            st.write("Epochs Analysis")
            distilcadiff_epochs = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\dist\epochs_accuracy.png')
            st.image(distilcadiff_epochs, caption="Epochs and Accuracy Analysis")
            
            st.markdown("---")  # Page break
            
            st.write("Batch Size Analysis")
            distilcadiff_batch = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\dist\batchsize_accuracy.png')
            st.image(distilcadiff_batch, caption="Batch Size and Accuracy Analysis")
        except:
            st.write("Distilcadiff metrics visualization not available")
        
        st.markdown("---")  # Page break
        
        st.subheader("Training Performance")
        try:
            distil_test_graph = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\dist\training_metrics_3_dist.png')
            st.image(distil_test_graph, caption="Accuracy and Loss Curves")
        except:
            st.write("Training metrics not available")
            
        st.markdown("---")  # Page break
        
        st.subheader("Test Results")
        try:
            distilcadiff_test_results = Image.open(r'C:\Users\11 PrO\Desktop\trying bots\dist\test_4.png')
            st.image(distilcadiff_test_results, caption="Best Performing Model's test Results")
            st.write("The model has an accuracy of 85%")
        except:
            st.write("Test results not available")
        
        st.markdown("---")  # Page break
        
        st.write("""
        ### Model Performance Summary
        - Architecture: distilbert-base-uncased
        - Fine-tuned on emotion dataset
        - Best validation accuracy achieved so far-85%
        - The Best Performing Model out of all the Models Experimented
        - Increase the epochs and your accuracy increases 
        - Weight Decay- greater than 0.01 lesser than or equal to 0.2
        - learning rate= Best accuracies when the value is around 5e-5

        """)

if __name__ == "__main__":
    main() 
