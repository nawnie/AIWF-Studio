export interface ControlNetCompatibilityLabelInput {
  supported: boolean
  controlNetFamily: 'sd15' | 'sdxl' | null
}

export function formatControlNetCompatibilityLabel(input: ControlNetCompatibilityLabelInput): string {
  if (!input.supported) return 'ControlNet unavailable'
  return input.controlNetFamily ? 'ControlNet family matches' : 'ControlNet compatibility unverified'
}
