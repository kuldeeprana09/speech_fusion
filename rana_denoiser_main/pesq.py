import os
import pesq

# Set the directory path
dir_path = '/path/to/wav/directory'

# Initialize PESQ object
pesq_obj = pesq()

# Loop over all WAV files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.wav'):
        file_path = os.path.join(dir_path, filename)
        
        # Load the reference (clean) and degraded (enhanced) signals
        clean_signal, fs_clean = pesq_obj.load_audio(file_path)
        enhanced_signal, fs_enhanced = pesq_obj.load_audio('enhanced_' + filename)
        
        # Match the lengths of the signals
        min_len = min(len(clean_signal), len(enhanced_signal))
        clean_signal = clean_signal[:min_len]
        enhanced_signal = enhanced_signal[:min_len]
        
        # Calculate the PESQ score
        score = pesq_obj.calc_pesq(clean_signal, enhanced_signal, fs_clean)
        
        # Print the score for the current file
        print(filename, score)