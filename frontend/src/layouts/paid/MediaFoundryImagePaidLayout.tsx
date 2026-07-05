import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Box,
  Check,
  ChevronDown,
  Download,
  Eye,
  EyeOff,
  Image as ImageIcon,
  Layers3,
  Library,
  Lock,
  Paintbrush,
  Plus,
  RefreshCcw,
  Search,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Wand2,
} from 'lucide-react'
import type { PaidLayoutProps } from './PaidLayoutTypes'
import { displayDate, selectedImage } from './PaidLayoutTypes'
import './paidLayouts.css'

const PAID_CANVAS_FALLBACK = '/paid-astronaut-canvas.png'

type DockMode = 'scenes' | 'tracks'
type InspectorMode = 'inspector' | 'metadata'

const EFFECTS = ['Color Grade', 'Curves', 'Contrast', 'Sharpen', 'Bloom']
const LAYERS = ['Color Grade', 'Sky Adjust', 'Astronaut', 'Mountains', 'Atmosphere', 'Lighting', 'Background']

export function MediaFoundryImagePaidLayout({
  settings,
  bootstrap,
  runtime,
  recentOutputs,
  preview,
  selectedModelName,
  statusMessage,
  isGenerating,
  onSettingsChange,
  onGenerate,
  onSendToWorkflow,
  onPreviewSelect,
  onOpenModels,
  onOpenSettings,
}: PaidLayoutProps) {
  const [dockMode, setDockMode] = useState<DockMode>('scenes')
  const [inspectorMode, setInspectorMode] = useState<InspectorMode>('inspector')
  const [activeLayer, setActiveLayer] = useState('Astronaut')
  const activeOutput = selectedImage(preview, recentOutputs)
  const variants = useMemo(() => recentOutputs.slice(0, 8), [recentOutputs])
  const assetRows = variants.length ? variants : activeOutput ? [activeOutput] : []
  const canvasUrl = activeOutput?.url || PAID_CANVAS_FALLBACK
  const canvasThumbUrl = activeOutput?.thumbnailUrl || PAID_CANVAS_FALLBACK

  return (
    <div className="paid-foundry paid-full-surface" aria-label="Media Foundry Image paid layout">
      <aside className="paid-foundry-assets">
        <div className="paid-product-lockup compact">
          <span className="paid-logo-orb">A</span>
          <div>
            <strong>AIWF Studio</strong>
            <small>Media Foundry · Image · PAID</small>
          </div>
        </div>
        <div className="paid-foundry-tabs">
          {['All', 'Images', 'Styles', 'Models', 'Prompts'].map((tab, index) => (
            <button key={tab} type="button" className={index === 0 ? 'active' : ''}>{tab}</button>
          ))}
        </div>
        <label className="paid-search-field">
          <Search size={14} aria-hidden="true" />
          <input value="" placeholder="Search assets..." readOnly />
        </label>
        <AssetSection title="Prompts" action="See all">
          {[settings.prompt || 'Astronaut explorer standing on a rocky outcrop...', 'Dramatic alien landscape at sunset, cinematic...', 'Futuristic city in the clouds, ultra detailed...'].map((prompt, index) => (
            <button key={prompt} type="button" className="paid-prompt-pill">
              <Sparkles size={13} />
              <span>{prompt}</span>
              <small>{index === 0 ? '2m ago' : `${index}h ago`}</small>
            </button>
          ))}
        </AssetSection>
        <AssetSection title="Source Images" action="See all">
          <div className="paid-thumb-grid">
            {assetRows.slice(0, 6).map((output) => (
              <button key={output.id} type="button" onClick={() => onPreviewSelect(output)}>
                <img src={output.thumbnailUrl} alt={output.prompt} />
              </button>
            ))}
          </div>
        </AssetSection>
        <AssetSection title="Styles" action="See all">
          <div className="paid-style-grid">
            {['Cinematic', 'Photoreal', 'Concept Art', 'Matte Painting', 'Moody', 'Sci-Fi'].map((style, index) => (
              <button key={style} type="button">
                {assetRows[index % Math.max(1, assetRows.length)] ? <img src={assetRows[index % Math.max(1, assetRows.length)].thumbnailUrl} alt="" /> : null}
                <span>{style}</span>
              </button>
            ))}
          </div>
        </AssetSection>
        <AssetSection title="Models" action="See all">
          {bootstrap.models.slice(0, 5).map((model, index) => (
            <button key={model.id} type="button" className="paid-model-row" onClick={onOpenModels}>
              <span>{index + 1}</span>
              <div>
                <strong>{model.name}</strong>
                <small>{model.architecture || model.backend || 'Local model'}</small>
              </div>
            </button>
          ))}
        </AssetSection>
      </aside>

      <main className="paid-foundry-stage">
        <header className="paid-foundry-topbar">
          <div className="paid-document-title">
            <ImageIcon size={15} />
            <div>
              <strong>{activeOutput?.prompt || 'Astronaut Explorer'}</strong>
              <small>{settings.width} × {settings.height} · sRGB · {statusMessage}</small>
            </div>
          </div>
          <div className="paid-foundry-top-actions">
            <button type="button"><RefreshCcw size={15} /> Undo</button>
            <button type="button">Redo</button>
            <button type="button" onClick={onOpenSettings}><Settings2 size={15} /></button>
            <button type="button" className="paid-export-button"><Download size={15} /> Export</button>
            <button type="button" onClick={() => onSendToWorkflow?.('Media Foundry')}><Sparkles size={15} /> Send to workflow</button>
          </div>
        </header>
        <div className="paid-tool-strip">
          {[
            ['Compare', Library],
            ['Inpaint', Paintbrush],
            ['Outpaint', Box],
            ['Upscale 2x', SlidersHorizontal],
            ['Face Fix', Wand2],
            ['Color Grade', SlidersHorizontal],
            ['Variants', Layers3],
          ].map(([label, Icon]) => {
            const ToolIcon = Icon as typeof Library
            return <button key={label as string} type="button"><ToolIcon size={14} />{label as string}</button>
          })}
        </div>
        <section className="paid-image-canvas">
          <div className="paid-floating-brushbar">
            <button type="button"><Paintbrush size={15} /></button>
            <button type="button"><Wand2 size={15} /></button>
            <button type="button">120 px</button>
            <label>Soft edge <input type="range" defaultValue="30" /></label>
            <label>Opacity <input type="range" defaultValue="100" /></label>
            <button type="button">Clear</button>
            <button type="button" className="done"><Check size={15} /> Done</button>
          </div>
          <div className="paid-canvas-frame">
            <img src={canvasUrl} alt={activeOutput?.prompt || 'Astronaut explorer'} />
            <div className="paid-canvas-minimap"><img src={canvasThumbUrl} alt="" /></div>
          </div>
        </section>

        <section className="paid-foundry-bottom-dock">
          <header>
            <div className="paid-dock-title">
              <strong>Session</strong>
              <small>{activeOutput?.prompt || 'No active scene'} · {displayDate(activeOutput?.createdAt || '')}</small>
            </div>
            <div className="paid-dock-tabs" role="tablist" aria-label="Bottom dock mode">
              <button type="button" className={dockMode === 'scenes' ? 'active' : ''} onClick={() => setDockMode('scenes')}>Scenes</button>
              <button type="button" className={dockMode === 'tracks' ? 'active' : ''} onClick={() => setDockMode('tracks')}>Tracks</button>
            </div>
            <button type="button" className="paid-dock-close"><ChevronDown size={16} /></button>
          </header>
          {dockMode === 'scenes' ? (
            <div className="paid-scene-board">
              <div className="paid-session-stats">
                <Stat label="Canvas" value={`${settings.width}×${settings.height}`} />
                <Stat label="Layers" value={`${LAYERS.length}`} />
                <Stat label="Model" value={selectedModelName.split(' ')[0] || 'Local'} />
                <Stat label="Steps" value={`${settings.steps}`} />
                <Stat label="CFG" value={`${settings.cfgScale}`} />
                <Stat label="Seed" value={`${settings.seed}`} />
              </div>
              <div className="paid-scene-strip">
                {variants.map((output, index) => (
                  <button key={output.id} type="button" className={preview?.id === output.id ? 'active' : ''} onClick={() => onPreviewSelect(output)}>
                    <img src={output.thumbnailUrl} alt={output.prompt} />
                    <strong>{index === 0 ? 'Base' : `V${index}`}</strong>
                    <small>{output.modelName || selectedModelName}</small>
                  </button>
                ))}
                <button type="button" className="paid-new-variant" onClick={onGenerate} disabled={isGenerating}>
                  <Plus size={22} />
                  New Variant
                </button>
                <button type="button" className="paid-new-variant" onClick={() => onSendToWorkflow?.('Media Foundry variant')}>
                  <Sparkles size={20} />
                  Send to workflow
                </button>
              </div>
            </div>
          ) : (
            <div className="paid-track-board">
              <div className="paid-track-ruler">
                {['00:00:00', '00:00:05', '00:00:10', '00:00:15', '00:00:20', '00:00:25', '00:00:30'].map((tick) => <span key={tick}>{tick}</span>)}
              </div>
              <TrackRow label="S3" title="Scene 3" color="violet" blocks={['Sky replace', 'Color grade', 'Export crop']} />
              <TrackRow label="S2" title="Scene 2" color="blue" blocks={['Inpaint pass', 'Upscale x2']} />
              <TrackRow label="S1" title="Scene 1" color="amber" blocks={['Base generation', 'Mask astronaut', 'VSR image']} />
              <TrackRow label="M1" title="Masks" color="white" blocks={['Mask 1', 'Mask 2']} />
              <TrackRow label="FX" title="Effects" color="green" blocks={['Curves', 'Sharpen', 'Bloom']} />
              <TrackRow label="MD" title="Metadata" color="slate" blocks={[`Prompt: ${settings.prompt.slice(0, 44) || 'Untitled'}`, `Model: ${selectedModelName.slice(0, 24)}`, `Seed: ${settings.seed}`]} />
              <div className="paid-playhead" />
            </div>
          )}
        </section>
      </main>

      <aside className="paid-foundry-inspector">
        <header className="paid-inspector-tabs">
          <button type="button" className={inspectorMode === 'inspector' ? 'active' : ''} onClick={() => setInspectorMode('inspector')}>Inspector</button>
          <button type="button" className={inspectorMode === 'metadata' ? 'active' : ''} onClick={() => setInspectorMode('metadata')}>Metadata</button>
          <button type="button" className="paid-export-button" onClick={onGenerate} disabled={isGenerating}>Run</button>
        </header>
        {inspectorMode === 'metadata' ? (
          <div className="paid-metadata-panel">
            <h3>Generation Metadata</h3>
            <dl>
              <div><dt>Model</dt><dd>{selectedModelName}</dd></div>
              <div><dt>Size</dt><dd>{settings.width}×{settings.height}</dd></div>
              <div><dt>Sampler</dt><dd>{settings.sampler}</dd></div>
              <div><dt>Seed</dt><dd>{settings.seed}</dd></div>
              <div><dt>Status</dt><dd>{runtime.state}</dd></div>
            </dl>
          </div>
        ) : (
          <>
            <InspectorSection title="Layers">
              <label className="paid-blend-row">Normal <input type="range" defaultValue="100" /> 100%</label>
              {LAYERS.map((layer, index) => (
                <button
                  key={layer}
                  type="button"
                  className={activeLayer === layer ? 'paid-layer-row active' : 'paid-layer-row'}
                  onClick={() => setActiveLayer(layer)}
                >
                  {index < 5 ? <Eye size={14} /> : <EyeOff size={14} />}
                  <span className="paid-layer-thumb">{assetRows[index % Math.max(1, assetRows.length)] ? <img src={assetRows[index % Math.max(1, assetRows.length)].thumbnailUrl} alt="" /> : null}</span>
                  <span>{layer}</span>
                  {layer === 'Background' ? <Lock size={12} /> : null}
                </button>
              ))}
            </InspectorSection>
            <InspectorSection title="Mask">
              <div className="paid-mask-card"><Wand2 size={18} /><span>Mask 2</span><label><input type="checkbox" /> Invert</label></div>
            </InspectorSection>
            <InspectorSection title="Effects Stack">
              {EFFECTS.map((effect, index) => (
                <label key={effect} className="paid-effect-row">
                  <input type="checkbox" defaultChecked />
                  <span>{effect}</span>
                  <input type="range" defaultValue={index === 3 ? 60 : 100} />
                  <output>{index === 3 ? '60' : '100'}</output>
                </label>
              ))}
              <button type="button" className="paid-add-effect"><Plus size={14} /> Add Effect</button>
              <button type="button" className="paid-add-effect" onClick={() => onSendToWorkflow?.('Media Foundry inspector')}>Send to workflow</button>
            </InspectorSection>
            <InspectorSection title="Generation Settings">
              <label>Model
                <select value={settings.modelId} onChange={(event) => onSettingsChange((current) => ({ ...current, modelId: event.target.value }))}>
                  {bootstrap.models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
                </select>
              </label>
              <label>Steps
                <input type="range" min="1" max="80" value={settings.steps} onChange={(event) => onSettingsChange((current) => ({ ...current, steps: Number(event.target.value) }))} />
              </label>
              <label>CFG Scale
                <input type="range" min="0" max="20" step="0.5" value={settings.cfgScale} onChange={(event) => onSettingsChange((current) => ({ ...current, cfgScale: Number(event.target.value) }))} />
              </label>
            </InspectorSection>
            <InspectorSection title="Prompt Recipe">
              <textarea value={settings.prompt} onChange={(event) => onSettingsChange((current) => ({ ...current, prompt: event.target.value }))} />
              <div className="paid-recipe-actions">
                <button type="button">Enhance</button>
                <button type="button">Expand</button>
              </div>
            </InspectorSection>
            <button type="button" className="paid-apply-button export" onClick={onGenerate} disabled={isGenerating}>Export Image</button>
          </>
        )}
      </aside>
    </div>
  )
}

function AssetSection({ title, action, children }: { title: string; action: string; children: ReactNode }) {
  return <section className="paid-asset-section"><header><h3>{title}</h3><button type="button">{action}</button></header>{children}</section>
}

function InspectorSection({ title, children }: { title: string; children: ReactNode }) {
  return <section className="paid-inspector-section"><header><h3>{title}</h3><button type="button">×</button></header>{children}</section>
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div className="paid-stat"><span>{label}</span><strong>{value}</strong></div>
}

function TrackRow({ label, title, color, blocks }: { label: string; title: string; color: string; blocks: string[] }) {
  return (
    <div className={`paid-track-row track-${color}`}>
      <div className="paid-track-label"><strong>{label}</strong><small>{title}</small></div>
      <div className="paid-track-lane">
        {blocks.map((block, index) => <span key={`${block}-${index}`} style={{ gridColumn: `${index * 3 + 1} / span ${index === blocks.length - 1 ? 3 : 4}` }}>{block}</span>)}
      </div>
    </div>
  )
}
