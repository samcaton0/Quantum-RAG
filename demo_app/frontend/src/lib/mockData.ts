/**
 * Mock data for demo when backend is unavailable
 */

import type { CompareResponse, MethodResult, UMAPPoint } from './api';

// Dataset-specific cluster configurations
const DATASET_CLUSTERS: Record<string, { name: string; cx: number; cy: number }[]> = {};

// Mock UMAP points for visualization
const generateMockUMAPPoints = (dataset: string): UMAPPoint[] => {
  const clusters = DATASET_CLUSTERS[dataset] || [];

  const points: UMAPPoint[] = [];
  let id = 0;

  clusters.forEach(cluster => {
    const count = 5 + Math.floor(Math.random() * 4);
    for (let i = 0; i < count; i++) {
      points.push({
        x: cluster.cx + (Math.random() - 0.5) * 1.5,
        y: cluster.cy + (Math.random() - 0.5) * 1.5,
        chunk_id: `chunk_${id++}`,
        source: `${cluster.name}_doc_${i + 1}.txt`,
        cluster: cluster.name,
        is_selected: false,
        selected_by: [],
      });
    }
  });

  return points;
};

// Dataset-specific mock results
const MOCK_DATA: Record<string, { topk: MethodResult; mmr: MethodResult; qubo: MethodResult }> = {
  wikipedia: {
    topk: {
      method: 'topk',
      results: [
        { rank: 1, score: 0.92, text: 'Infectious mononucleosis typically presents with fatigue, sore throat, and swollen lymph nodes. Patients often experience extreme tiredness lasting several weeks.', source: 'mononucleosis_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.91, text: 'Mono symptoms include persistent fatigue, fever, and joint pain. The Epstein-Barr virus is the most common cause of infectious mononucleosis.', source: 'mononucleosis_doc_2.txt', chunk_id: 'chunk_1' },
        { rank: 3, score: 0.89, text: 'Mononucleosis causes significant fatigue and malaise. Patients may experience muscle aches and low-grade fever for extended periods.', source: 'mononucleosis_doc_3.txt', chunk_id: 'chunk_2' },
        { rank: 4, score: 0.88, text: 'The glandular fever virus leads to chronic tiredness and joint discomfort. Recovery from mononucleosis can take several months.', source: 'mononucleosis_doc_4.txt', chunk_id: 'chunk_3' },
        { rank: 5, score: 0.87, text: 'EBV infection manifests as extreme exhaustion with occasional febrile episodes. Splenic enlargement may occur in severe cases.', source: 'mononucleosis_doc_5.txt', chunk_id: 'chunk_4' },
      ],
      metrics: { latency_ms: 23, intra_list_similarity: 0.72, cluster_coverage: 1, total_clusters: 5, avg_relevance: 0.894 },
      llm_response: 'Based on the retrieved documents, the symptoms suggest infectious mononucleosis (mono). The patient should be tested for Epstein-Barr virus. Note: This response is limited because all retrieved documents discuss only mononucleosis.',
    },
    mmr: {
      method: 'mmr',
      results: [
        { rank: 1, score: 0.92, text: 'Infectious mononucleosis typically presents with fatigue, sore throat, and swollen lymph nodes. Patients often experience extreme tiredness lasting several weeks.', source: 'mononucleosis_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.78, text: 'Systemic lupus erythematosus (SLE) causes fatigue, joint pain, and low-grade fever. Autoimmune inflammation affects multiple organ systems.', source: 'lupus_doc_1.txt', chunk_id: 'chunk_10' },
        { rank: 3, score: 0.75, text: 'Lyme disease from tick bites causes fatigue, joint pain, and fever. Early treatment with antibiotics is essential.', source: 'lyme_doc_1.txt', chunk_id: 'chunk_15' },
        { rank: 4, score: 0.88, text: 'Mono symptoms include persistent fatigue, fever, and joint pain. The Epstein-Barr virus is the most common cause.', source: 'mononucleosis_doc_2.txt', chunk_id: 'chunk_1' },
        { rank: 5, score: 0.71, text: 'Chronic fatigue syndrome presents with debilitating tiredness not improved by rest. Cognitive difficulties are common.', source: 'chronic_fatigue_doc_1.txt', chunk_id: 'chunk_25' },
      ],
      metrics: { latency_ms: 45, intra_list_similarity: 0.48, cluster_coverage: 4, total_clusters: 5, avg_relevance: 0.808 },
      llm_response: 'The symptoms could indicate several conditions: mononucleosis, lupus, Lyme disease, or chronic fatigue syndrome. Further testing is recommended to differentiate between these diagnoses.',
    },
    qubo: {
      method: 'qubo',
      results: [
        { rank: 1, score: 0.92, text: 'Infectious mononucleosis typically presents with fatigue, sore throat, and swollen lymph nodes. Patients often experience extreme tiredness lasting several weeks.', source: 'mononucleosis_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.78, text: 'Systemic lupus erythematosus (SLE) causes fatigue, joint pain, and low-grade fever. Characteristic butterfly rash and photosensitivity may be present.', source: 'lupus_doc_1.txt', chunk_id: 'chunk_10' },
        { rank: 3, score: 0.75, text: 'Lyme disease from tick bites causes fatigue, joint pain, and fever. Bulls-eye rash (erythema migrans) is a distinctive early sign.', source: 'lyme_doc_1.txt', chunk_id: 'chunk_15' },
        { rank: 4, score: 0.73, text: 'Fibromyalgia causes widespread musculoskeletal pain with fatigue, sleep problems, and cognitive difficulties. Tender points are characteristic.', source: 'fibromyalgia_doc_1.txt', chunk_id: 'chunk_20' },
        { rank: 5, score: 0.71, text: 'Chronic fatigue syndrome presents with debilitating tiredness not improved by rest. Post-exertional malaise is a key diagnostic criterion.', source: 'chronic_fatigue_doc_1.txt', chunk_id: 'chunk_25' },
      ],
      metrics: { latency_ms: 87, intra_list_similarity: 0.31, cluster_coverage: 5, total_clusters: 5, avg_relevance: 0.778 },
      llm_response: 'The symptoms warrant evaluation for multiple conditions:\n\n1. **Mononucleosis** - Check for EBV antibodies\n2. **Lupus** - ANA test, look for butterfly rash\n3. **Lyme disease** - Check for tick exposure, order Lyme titers\n4. **Fibromyalgia** - Assess tender points\n5. **Chronic fatigue syndrome** - Rule out other causes first\n\nComprehensive differential diagnosis enables targeted testing.',
    },
  },
};

export function getMockCompareResponse(query: string, dataset: string): CompareResponse {
  const datasetKey = dataset in MOCK_DATA ? dataset : 'wikipedia';
  const mockData = MOCK_DATA[datasetKey];
  const umapPoints = generateMockUMAPPoints(datasetKey);

  // Mark selected points based on mock results
  const topkIds = mockData.topk.results.map(r => r.chunk_id);
  const mmrIds = mockData.mmr.results.map(r => r.chunk_id);
  const quboIds = mockData.qubo.results.map(r => r.chunk_id);

  umapPoints.forEach(point => {
    const selectedBy: string[] = [];
    if (topkIds.includes(point.chunk_id)) selectedBy.push('topk');
    if (mmrIds.includes(point.chunk_id)) selectedBy.push('mmr');
    if (quboIds.includes(point.chunk_id)) selectedBy.push('qubo');

    if (selectedBy.length > 0) {
      point.is_selected = true;
      point.selected_by = selectedBy;
    }
  });

  return {
    query,
    dataset,
    topk: mockData.topk,
    mmr: mockData.mmr,
    qubo: mockData.qubo,
    umap_points: umapPoints,
    query_point: { x: 0.5, y: 0.5 },
  };
}

export const DEMO_QUERIES = {
  wikipedia: [
    'How do greenhouse gases and carbon emissions contribute to climate change, and what renewable energy solutions can mitigate environmental damage?',
    'Explain quantum mechanical principles and their applications in modern quantum computing and field theory',
    'What are the foundational technologies behind artificial intelligence, machine learning, and neural networks, and how do they enable modern robotics?',
    'How does the immune system respond to diseases like COVID-19 and what role do vaccines play in public health?',
    'Trace the evolution of democracy, human rights, and major civil rights movements including women\'s suffrage and feminism',
    'Evaluate modern transportation systems including rail transit, electric vehicles, and high-speed infrastructure',
  ],
};
