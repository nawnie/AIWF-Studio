import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Eye,
  EyeOff,
  FolderOpen,
  Layers3,
  Plug,
  Save,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { LayoutProps, TabsProps } from './LayoutTypes'
import { saveUserTabs } from './studioApiClient'
import './studioLayouts.css'

type SettingSection = 'generation' | 'samplers' | 'hires' | 'inpaint' | 'control' | 'lora' | 'output' | 'paths' | 'ui-tabs' | 'agent'

const SECTIONS: Array<{ id: SettingSection; label: string; hint: string }> = [
  { id: 'generation', label: 'Generation', hint: 'Model, size, seed, batch, prompt defaults' },
  { id: 'samplers', label: 'Samplers', hint: 'Sampler, scheduler, steps, CFG, clip skip' },
  { id: 'hires', label: 'Hires / Refiner', hint: 'Hires pass, upscale, denoise, refiner controls' },
  { id: 'inpaint', label: 'Inpaint', hint: 'Mask content, blur, padding, masked only' },
  { id: 'control', label: 'ControlNet', hint: 'Units, preprocessors, weights, pixel perfect' },
  { id: 'lora', label: 'LoRA / Extra Nets', hint: 'Trigger words, stacks, fuse/export helpers' },
  { id: 'output', label: 'Output', hint: 'Format, metadata, filename, receipts' },
  { id: 'paths', label: 'Paths', hint: 'Models, outputs, engines, external SDKs' },
  { id: 'ui-tabs', label: 'UI Tabs', hint: 'Show/hide built-in and user-added tabs' },
  { id: 'agent', label: 'Agent', hint: 'Ollama, skills, tool permissions' },
]

export function SettingsArsenalLayout({
  settings,
  bootstrap,
  runtime,
  selectedModelName,
  statusMessage,
  onSettingsChange,
  userTabs,
  onTabsChange,
}: LayoutProps & TabsProps) {
  const [activeSection, setActiveSection] = useState<SettingSection>('generation')
  const visibleTabs = useMemo(() => userTabs.filter((tab) => !tab.hidden), [userTabs])
  const updateTabs = (nextTabs: typeof userTabs) => {
    onTabsChange(nextTabs)
    void saveUserTabs(nextTabs)
  }

  return (
    <div className="studio-settings studio-full-surface" aria-label="Advanced paid settings layout">
      <aside className="studio-settings-nav">
        <div className="studio-product-lockup compact">
          <span className="studio-logo-orb"><Settings2 size={20} /></span>
          <div>
            <strong>Settings Arsenal</strong>
            <small>A1111-style depth</small>
          </div>
        </div>
        {SECTIONS.map((section) => (
          <button key={section.id} type="button" className={activeSection === section.id ? 'active' : ''} onClick={() => setActiveSection(section.id)}>
            <span>{section.label}</span>
            <small>{section.hint}</small>
          </button>
        ))}
      </aside>

      <main className="studio-settings-main">
        <header className="studio-agent-header">
          <div>
            <span className="studio-eyebrow">ADVANCED SETTINGS</span>
            <strong>{SECTIONS.find((section) => section.id === activeSection)?.label}</strong>
            <small>{runtime.state} · {selectedModelName} · {statusMessage}</small>
          </div>
          <button type="button" className="studio-run-button"><Save size={15} /> Save Settings</button>
        </header>

        {activeSection === 'generation' ? (
          <SettingsGrid>
            <Field label="Prompt default"><textarea value={settings.prompt} rows={4} onChange={(event) => onSettingsChange((current) => ({ ...current, prompt: event.target.value }))} /></Field>
            <Field label="Negative prompt"><textarea value={settings.negativePrompt} rows={4} onChange={(event) => onSettingsChange((current) => ({ ...current, negativePrompt: event.target.value }))} /></Field>
            <Field label="Checkpoint"><select value={settings.modelId} onChange={(event) => onSettingsChange((current) => ({ ...current, modelId: event.target.value }))}>{bootstrap.models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select></Field>
            <Field label="Width"><input type="number" value={settings.width} onChange={(event) => onSettingsChange((current) => ({ ...current, width: Number(event.target.value) }))} /></Field>
            <Field label="Height"><input type="number" value={settings.height} onChange={(event) => onSettingsChange((current) => ({ ...current, height: Number(event.target.value) }))} /></Field>
            <Field label="Batch size"><input type="number" min="1" max="4" value={settings.batchSize} onChange={(event) => onSettingsChange((current) => ({ ...current, batchSize: Number(event.target.value) }))} /></Field>
            <Field label="Batch count"><input type="number" min="1" max="16" value={settings.batchCount} onChange={(event) => onSettingsChange((current) => ({ ...current, batchCount: Number(event.target.value) }))} /></Field>
            <Field label="Seed"><input type="number" value={settings.seed} onChange={(event) => onSettingsChange((current) => ({ ...current, seed: Number(event.target.value) }))} /></Field>
          </SettingsGrid>
        ) : activeSection === 'samplers' ? (
          <SettingsGrid>
            <Field label="Sampler"><input value={settings.sampler} onChange={(event) => onSettingsChange((current) => ({ ...current, sampler: event.target.value }))} /></Field>
            <Field label="Scheduler"><input value={settings.scheduler} onChange={(event) => onSettingsChange((current) => ({ ...current, scheduler: event.target.value }))} /></Field>
            <Range label="Steps" value={settings.steps} min={1} max={150} onChange={(value) => onSettingsChange((current) => ({ ...current, steps: value }))} />
            <Range label="CFG scale" value={settings.cfgScale} min={0} max={30} step={0.5} onChange={(value) => onSettingsChange((current) => ({ ...current, cfgScale: value }))} />
            <Range label="Clip skip" value={settings.clipSkip} min={1} max={12} onChange={(value) => onSettingsChange((current) => ({ ...current, clipSkip: value }))} />
            <InfoCard icon={SlidersHorizontal} title="A1111 parity lane" text="Sampler aliases, schedule type, ETA, sigma controls, and variation seed are reserved here for the next backend settings pass." />
          </SettingsGrid>
        ) : activeSection === 'hires' ? (
          <SettingsGrid>
            <CheckField label="Enable hires fix" checked={settings.enableHires} onChange={(value) => onSettingsChange((current) => ({ ...current, enableHires: value }))} />
            <Range label="Hires scale" value={settings.hiresScale} min={1} max={4} step={0.05} onChange={(value) => onSettingsChange((current) => ({ ...current, hiresScale: value }))} />
            <Range label="Hires steps" value={settings.hiresSteps} min={1} max={150} onChange={(value) => onSettingsChange((current) => ({ ...current, hiresSteps: value }))} />
            <Range label="Hires denoise" value={settings.hiresDenoise} min={0} max={1} step={0.01} onChange={(value) => onSettingsChange((current) => ({ ...current, hiresDenoise: value }))} />
            <Field label="Hires upscaler"><input value={settings.hiresUpscaler} onChange={(event) => onSettingsChange((current) => ({ ...current, hiresUpscaler: event.target.value }))} /></Field>
            <InfoCard icon={Layers3} title="Refiner slot" text="SDXL refiner checkpoint, refiner switch step, and refiner denoise controls belong here when the backend exposes them to Pro." />
          </SettingsGrid>
        ) : activeSection === 'inpaint' ? (
          <SettingsGrid>
            <Range label="Denoising strength" value={settings.denoisingStrength} min={0} max={1} step={0.01} onChange={(value) => onSettingsChange((current) => ({ ...current, denoisingStrength: value }))} />
            <Range label="Mask blur" value={settings.maskBlur} min={0} max={64} onChange={(value) => onSettingsChange((current) => ({ ...current, maskBlur: value }))} />
            <Range label="Masked padding" value={settings.inpaintMaskedPadding} min={0} max={256} onChange={(value) => onSettingsChange((current) => ({ ...current, inpaintMaskedPadding: value }))} />
            <Range label="Mask opacity" value={settings.inpaintMaskOpacity} min={0} max={1} step={0.01} onChange={(value) => onSettingsChange((current) => ({ ...current, inpaintMaskOpacity: value }))} />
            <CheckField label="Inpaint only masked" checked={settings.inpaintOnlyMasked} onChange={(value) => onSettingsChange((current) => ({ ...current, inpaintOnlyMasked: value }))} />
            <Field label="Masked content"><input value={settings.inpaintMaskContent} onChange={(event) => onSettingsChange((current) => ({ ...current, inpaintMaskContent: event.target.value }))} /></Field>
          </SettingsGrid>
        ) : activeSection === 'control' ? (
          <PlaceholderSection icon={ShieldCheck} title="ControlNet / conditioning" rows={['Multi-unit controls', 'Preprocessor selection', 'Pixel-perfect size', 'Control weight and guidance window', 'Reference-only and IP adapter lanes']} />
        ) : activeSection === 'lora' ? (
          <PlaceholderSection icon={Sparkles} title="LoRA / Extra Networks" rows={['LoRA stack slots', 'Trigger word insertion', 'Saved alias and strength', 'Architecture compatibility warning', 'Fuse/export worker']} />
        ) : activeSection === 'output' ? (
          <SettingsGrid>
            <Field label="Image format"><select defaultValue="png"><option>png</option><option>jpg</option><option>webp</option></select></Field>
            <CheckField label="Save images" checked={settings.saveImages} onChange={(value) => onSettingsChange((current) => ({ ...current, saveImages: value }))} />
            <InfoCard icon={Save} title="Receipt-first output" text="Filename pattern, sidecar text, PNG metadata, model hashes, LoRA hashes, optimization profile, and interrupted output controls live here." />
          </SettingsGrid>
        ) : activeSection === 'paths' ? (
          <PlaceholderSection icon={FolderOpen} title="Paths and engines" rows={['Models root', 'Checkpoint root', 'Extra model libraries', 'Outputs root', 'NVIDIA VideoFX SDK', 'Audio engine venv', 'Ollama URL']} />
        ) : activeSection === 'ui-tabs' ? (
          <section className="studio-settings-card-grid">
            <InfoCard icon={Plug} title="User-added tabs" text={`${userTabs.length} community tab(s), ${visibleTabs.length} visible in the left rail.`} />
            {userTabs.map((tab) => (
              <article key={tab.id} className="studio-settings-card">
                <header><Plug size={18} /><strong>{tab.label}</strong></header>
                <p>{tab.description || tab.id}</p>
                <button type="button" onClick={() => updateTabs(userTabs.map((item) => item.id === tab.id ? { ...item, hidden: !item.hidden } : item))}>
                  {tab.hidden ? <Eye size={14} /> : <EyeOff size={14} />}
                  {tab.hidden ? 'Show in rail' : 'Hide from rail'}
                </button>
              </article>
            ))}
          </section>
        ) : (
          <PlaceholderSection icon={Sparkles} title="Agentic assistant" rows={['Ollama model URL', 'Allowed tool list', 'Skill registry', 'Plugin permissions', 'Patch draft mode', 'Human confirmation gates']} />
        )}
      </main>
    </div>
  )
}

function SettingsGrid({ children }: { children: ReactNode }) {
  return <section className="studio-settings-grid">{children}</section>
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return <label className="studio-settings-field"><span>{label}</span>{children}</label>
}

function Range({ label, value, min, max, step = 1, onChange }: { label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void }) {
  return <label className="studio-settings-field range"><span>{label}</span><input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} /><b>{value}</b></label>
}

function CheckField({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return <label className="studio-settings-field check"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span>{label}</span></label>
}

function InfoCard({ icon: Icon, title, text }: { icon: LucideIcon; title: string; text: string }) {
  return <article className="studio-settings-card"><header><Icon size={18} /><strong>{title}</strong></header><p>{text}</p></article>
}

function PlaceholderSection({ icon: Icon, title, rows }: { icon: LucideIcon; title: string; rows: string[] }) {
  return <section className="studio-settings-card-grid"><InfoCard icon={Icon} title={title} text="Reserved UI slots are present now; backend fields can be wired one-by-one without layout churn." />{rows.map((row) => <article key={row} className="studio-settings-card"><strong>{row}</strong><p>Visible control slot ready for typed backend wiring.</p></article>)}</section>
}
