export interface AudioArtifactLicense {
  model_id: string
  checked: string
  license: string
  commercial: 'yes' | 'conditional' | 'no'
  conditions: string
  provenance: string
  source: string
  family: string
}

export interface AudioArtifactResult {
  modelId: string
  license?: AudioArtifactLicense
}

export interface AudioProjectAttribution {
  modelId: string
  license: AudioArtifactLicense | null
  licenseNotice: string | null
}

export function audioProjectAttribution(
  artifact: AudioArtifactResult | null | undefined,
  selectedModelId: string,
): AudioProjectAttribution {
  const modelId = artifact?.modelId || selectedModelId
  const license = artifact?.license
  if (!artifact || !license || !license.model_id) {
    return { modelId, license: null, licenseNotice: null }
  }
  if (license.model_id !== artifact.modelId) {
    throw new Error('The audio artifact license does not match its generating model.')
  }
  const parts = [
    license.license,
    `commercial use: ${license.commercial}`,
    ...(license.checked ? [`checked: ${license.checked}`] : []),
    ...(license.conditions ? [`conditions: ${license.conditions}`] : []),
    ...(license.source ? [`source: ${license.source}`] : []),
  ]
  return { modelId, license, licenseNotice: parts.join('; ') }
}
