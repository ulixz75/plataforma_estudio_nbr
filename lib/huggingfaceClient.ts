import { HfInference } from "@huggingface/inference";
import getInstructions from "./getInstructions";
import { ChatHistoryType } from "@/@types/appContext";

const hf = new HfInference(process.env.NEXT_PUBLIC_HUGGINGFACE_TOKEN || '');

const formatPrompt = (personality: TeacherPersonality, message: string, history?: ChatHistoryType[]): string => {
    const systemInstruction = getInstructions(personality);

    const historyText = history
        ? history.map(turn => `${turn.role}: ${turn.parts[0].text}`).join('\n')
        : '';

    // A simple prompt format. This might need to be adjusted depending on the model.
    // For many models, a structured format like Llama2's chat template works best.
    // But for a generic starting point, this is reasonable.
    return `${systemInstruction}\n\n${historyText}\nuser: ${message}\nmodel:`;
};

const interactionHuggingFace = async (message: string, personality: TeacherPersonality, history?: ChatHistoryType[]) => {
    const prompt = formatPrompt(personality, message, history);

    try {
        const response = await hf.textGeneration({
            model: process.env.NEXT_PUBLIC_HUGGINGFACE_MODEL || '',
            inputs: prompt,
            parameters: {
                max_new_tokens: 2048,
                temperature: 0.7,
                top_p: 0.95,
            }
        });

        // We wrap the response to mimic the structure of the Gemini API response
        // to minimize changes in the rest of the application.
        return {
            text: () => response.generated_text,
        };
    } catch (error) {
        console.error("Error interacting with Hugging Face API:", error);
        // Return a response structure that is consistent with expected success case to avoid crashes downstream
        return {
            text: () => "Lo siento, ha ocurrido un error al contactar al modelo. Por favor, intenta de nuevo."
        };
    }
};

export default interactionHuggingFace;