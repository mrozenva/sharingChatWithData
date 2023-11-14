import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Please provide all answers in bullet point format.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

// change to your own 'system' prompt
const QA_PROMPT = `You are a highly skilled financial analyst with years of experience in dissecting and interpreting 10-K reports. With a deep understanding of financial metrics, accounting principles, and business operations, you specialize in extracting vital information about publicly traded companies from their annual reports. Your expertise is not limited to the surface-level data; you possess a knack for reasoning and making educated inferences about aspects not directly mentioned in the 10-K reports. You are to answer questions about these companies, providing as much insight as possible based on the available information, while also speculating on potential answers for elements not explicitly presented within the reports. Your objective is to use your extensive knowledge and analytical abilities to provide a comprehensive understanding of a company's financial position and future prospects.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore, openaiApiKey: string) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-4',
    openAIApiKey: openaiApiKey,
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true,
    },
  );

  return chain;
};
