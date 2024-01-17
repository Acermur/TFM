from django.shortcuts import render
from joblib import load
import os
from django.conf import settings

# Cargar el modelo SVM y el escalador
svm_model_path = os.path.join(settings.BASE_DIR, 'modelos', 'optimized_svm_model.joblib')
scaler_path = os.path.join(settings.BASE_DIR, 'modelos', 'scaler.joblib')

try:
    svm_model = load(svm_model_path)
    scaler = load(scaler_path)
except Exception as e:
    print(f"Error loading SVM model: {e}")



class_names = {
    1: 'psoriasis',
    2: 'seborrheic dermatitis',
    3: 'lichen planus',
    4: 'pityriasis rosea',
    5: 'chronic dermatitis',
    6: 'pityriasis rubra pilaris'}

    
def home(request):
    return render(request, 'home.html')


def svm_predict(request):
    if request.method == 'POST':
        # datos del formulario
        features = [
            request.POST.get('erythema'),
            request.POST.get('scaling'),
            request.POST.get('definite_borders'),
            request.POST.get('itching'),
            request.POST.get('koebner_phenomenon'),
            request.POST.get('polygonal_papules'),
            request.POST.get('follicular_papules'),
            request.POST.get('oral_mucosal_involvement'),
            request.POST.get('knee_and_elbow_involvement'),
            request.POST.get('scalp_involvement'),
            request.POST.get('family_history'),
            request.POST.get('melanin_incontinence'),
            request.POST.get('eosinophils_in_the_infiltrate'),
            request.POST.get('PNL_infiltrate'),
            request.POST.get('fibrosis_of_the_papillary_dermis'),
            request.POST.get('exocytosis'),
            request.POST.get('acanthosis'),
            request.POST.get('hyperkeratosis'),
            request.POST.get('parakeratosis'),
            request.POST.get('clubbing_of_the_rete_ridges'),
            request.POST.get('elongation_of_the_rete_ridges'),
            request.POST.get('thinning_of_the_suprapapillary_epidermis'),
            request.POST.get('spongiform_pustule'),
            request.POST.get('munro_microabcess'),
            request.POST.get('focal_hypergranulosis'),
            request.POST.get('disappearance_of_the_granular_layer'),
            request.POST.get('vacuolisation_and_damage_of_basal_layer'),
            request.POST.get('spongiosis'),
            request.POST.get('saw_tooth_appearance_of_retes'),
            request.POST.get('follicular_horn_plug'),
            request.POST.get('perifollicular_parakeratosis'),
            request.POST.get('inflammatory_monoluclear_inflitrate'),
            request.POST.get('band_like_infiltrate'),
            request.POST.get('age'),
        ]
        # Validar y convertir los datos
        try:
            features = [float(x) if x is not None else 0.0 for x in features]
        except ValueError:
            return render(request, 'formulario.html', {'error': 'Invalid input'})

        # Preprocesar los datos 
        features_scaled = scaler.transform([features])

        # Realizar la predicción
        prediction = svm_model.predict(features_scaled)
        prediction_class = class_names.get(prediction[0], "Unknown disease")
        # Enviar la predicción a la plantilla de resultado
        return render(request, 'resultado.html', {'prediction': prediction_class})

    # Mostrar el formulario 
    return render(request, 'formulario.html')
