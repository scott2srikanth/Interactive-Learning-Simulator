import React from 'react';
import { architectureTemplates } from '../../lib/architectureTemplates';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Layers, Zap } from 'lucide-react';

interface TemplateSelectorProps {
  onSelectTemplate: (templateId: string) => void;
}

export const TemplateSelector: React.FC<TemplateSelectorProps> = ({ onSelectTemplate }) => {
  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <Zap className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Quick Start Templates</h3>
        </div>
        <p className="text-sm text-gray-600">
          Start with a pre-built architecture or design your own from scratch
        </p>

        <div className="space-y-3">
          {architectureTemplates.map((template) => (
            <div
              key={template.id}
              className="border border-gray-200 rounded-lg p-4 hover:border-blue-500 hover:bg-blue-50 transition-all"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <Layers className="w-4 h-4 text-gray-600" />
                    <h4 className="font-semibold text-gray-900">{template.name}</h4>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{template.description}</p>
                  <p className="text-xs text-gray-500 mt-2">
                    {template.layers.length} layers
                  </p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => onSelectTemplate(template.id)}
                >
                  Load
                </Button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};
