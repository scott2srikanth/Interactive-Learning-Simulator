import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { CNNBuilder } from '../components/cnn/CNNBuilder';
import { TrainingDashboard } from '../components/cnn/TrainingDashboard';
import { FeatureMapGrid, RGBImageCanvas } from '../components/visualization/FeatureMapGrid';
import { KernelEditor } from '../components/cnn/KernelEditor';
import { ArchitectureList } from '../components/cnn/ArchitectureList';
import { TemplateSelector } from '../components/cnn/TemplateSelector';
import { useCNNStore } from '../store/cnnStore';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { sampleDataset } from '../lib/datasets';
import { architectureTemplates } from '../lib/architectureTemplates';
import { Matrix2D } from '../types/cnn';
import { convolve2D } from '../lib/cnn/convolution';
import { Image, ArrowLeft } from 'lucide-react';
import { TOPICS } from '../types/topics';

export const Lab: React.FC = () => {
  const { topicId } = useParams();
  const navigate = useNavigate();
  const currentTopic = TOPICS.find(t => t.id === topicId);

  if (topicId !== 'cnn') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <Card className="max-w-md">
          <div className="p-8 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">Coming Soon</h2>
            <p className="text-slate-300 mb-6">
              The {currentTopic?.name} lab is under development. Check back soon!
            </p>
            <Button onClick={() => navigate('/topics')}>
              Back to Topics
            </Button>
          </div>
        </Card>
      </div>
    );
  }
  const { setInput, layerOutputs, architecture, addLayer, clearArchitecture, runForwardPass } = useCNNStore();
  const [selectedImage, setSelectedImage] = useState(sampleDataset[0]);
  const [activeTab, setActiveTab] = useState<'builder' | 'training' | 'visualizer'>('builder');
  const [customKernel, setCustomKernel] = useState<Matrix2D>([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ]);
  const [kernelOutput, setKernelOutput] = useState<Matrix2D | null>(null);

  const isColorImage = selectedImage.data.length === 3;

  const handleImageSelect = (imageId: string) => {
    const image = sampleDataset.find((img) => img.id === imageId);
    if (image) {
      setSelectedImage(image);
      setInput(image.data);
    }
  };

  const handleLoadTemplate = async (templateId: string) => {
    const template = architectureTemplates.find((t) => t.id === templateId);
    if (template) {
      clearArchitecture();

      setTimeout(() => {
        setInput(selectedImage.data);

        template.layers.forEach((layer) => {
          addLayer({
            ...layer,
            id: `${layer.type}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          });
        });

        setTimeout(async () => {
          await runForwardPass();
        }, 300);
      }, 50);

      setActiveTab('builder');
    }
  };

  const handleKernelChange = (newKernel: Matrix2D) => {
    setCustomKernel(newKernel);
    if (selectedImage) {
      const result = convolve2D(selectedImage.data[0], newKernel, 1);
      setKernelOutput(result);
    }
  };

  React.useEffect(() => {
    if (selectedImage) {
      setInput(selectedImage.data);
      const result = convolve2D(selectedImage.data[0], customKernel, 1);
      setKernelOutput(result);
    }
  }, [selectedImage, setInput]);


  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                onClick={() => navigate('/topics')}
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Topics
              </Button>
              <h1 className="text-2xl font-bold text-gray-900">CNN Lab</h1>
            </div>
            <div className="flex space-x-2">
              <Button
                variant={activeTab === 'builder' ? 'primary' : 'ghost'}
                onClick={() => setActiveTab('builder')}
              >
                Architecture Builder
              </Button>
              <Button
                variant={activeTab === 'training' ? 'primary' : 'ghost'}
                onClick={() => setActiveTab('training')}
              >
                Training
              </Button>
              <Button
                variant={activeTab === 'visualizer' ? 'primary' : 'ghost'}
                onClick={() => setActiveTab('visualizer')}
              >
                Kernel Visualizer
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <Card title="Input Image">
              <div className="space-y-4">
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <Image className="w-4 h-4" />
                  <span>Select an image:</span>
                </div>
                <select
                  value={selectedImage.id}
                  onChange={(e) => handleImageSelect(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {sampleDataset.map((img) => (
                    <option key={img.id} value={img.id}>
                      {img.name}
                    </option>
                  ))}
                </select>
                {selectedImage && (
                  <div className="mt-4">
                    {isColorImage ? (
                      <RGBImageCanvas data={selectedImage.data} label={selectedImage.label} />
                    ) : (
                      <>
                        <FeatureMapGrid data={selectedImage.data} />
                        <p className="text-sm text-gray-600 mt-2">Label: {selectedImage.label}</p>
                      </>
                    )}
                  </div>
                )}
              </div>
            </Card>

            {architecture.layers.length === 0 && (
              <TemplateSelector onSelectTemplate={handleLoadTemplate} />
            )}

            {architecture.layers.length > 0 && (
              <Card title="Model Info">
                <div className="space-y-2 text-sm mb-4">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Layers:</span>
                    <span className="font-semibold">{architecture.layers.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Parameters:</span>
                    <span className="font-semibold">
                      {useCNNStore.getState().getTotalParameters().toLocaleString()}
                    </span>
                  </div>
                </div>
                <div className="border-t border-gray-200 pt-4">
                  <ArchitectureList />
                </div>
              </Card>
            )}
          </div>

          <div className="lg:col-span-3">
            {activeTab === 'builder' && (
              <Card className="h-[700px]">
                <CNNBuilder />
              </Card>
            )}

            {activeTab === 'training' && <TrainingDashboard />}

            {activeTab === 'visualizer' && (
              <div className="space-y-6">
                <Card>
                  <KernelEditor kernel={customKernel} onChange={handleKernelChange} />
                </Card>

                {kernelOutput && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card title="Input Image">
                      {isColorImage ? (
                        <RGBImageCanvas data={selectedImage.data} label={selectedImage.label} />
                      ) : (
                        <FeatureMapGrid data={selectedImage.data} />
                      )}
                    </Card>
                    <Card title="After Convolution">
                      <FeatureMapGrid data={kernelOutput} />
                    </Card>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
