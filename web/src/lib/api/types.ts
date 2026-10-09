// Every response type is generated from contracts/openapi.json (npm run gen:types); never written by hand.
import type { components } from "./schema";

type C = components["schemas"];

export type Health = C["Health"];
export type FeatureList = C["FeatureList"];
export type FeatureListItem = C["FeatureListItem"];
export type FeatureDetail = C["FeatureDetail"];
export type Example = C["Example"];
export type LogitEntry = C["LogitEntry"];
export type ErrorBody = C["ErrorBody"];
export type LabelStatus = FeatureListItem["label_status"];

export type InspectRequest = C["InspectRequest"];
export type InspectResponse = C["InspectResponse"];
export type InspectToken = C["InspectToken"];
export type TokenFeature = C["TokenFeature"];
export type SteerRequest = C["SteerRequest"];
export type SteerResponse = C["SteerResponse"];
export type SteerMode = SteerRequest["mode"];
export type SteeringExample = C["SteeringExample"];
export type SteeringExampleList = C["SteeringExampleList"];
export type Run = C["Run"];
export type RunList = C["RunList"];
export type MetricsResponse = C["MetricsResponse"];
export type CurvePoint = C["CurvePoint"];
export type DensityBin = C["DensityBin"];

// Phase 4 steering checks (X1 to X4)
export type SteerCheckResponse = C["SteerCheckResponse"];
export type SteerCompareResponse = C["SteerCompareResponse"];
export type SteerSweepRequest = C["SteerSweepRequest"];
export type SteerSweepResponse = C["SteerSweepResponse"];
export type SweepResult = C["SweepResult"];
export type PromptOnlyRequest = C["PromptOnlyRequest"];
export type PromptOnlyResponse = C["PromptOnlyResponse"];

// Phase 4: X6 sentence report, X7 two-text compare, W4 co-fire partners
export type ReportRequest = C["ReportRequest"];
export type ReportResponse = C["ReportResponse"];
export type ReportSentence = C["ReportSentence"];
export type ReportFeature = C["ReportFeature"];
export type TextCompareRequest = C["TextCompareRequest"];
export type TextCompareResponse = C["TextCompareResponse"];
export type TextCompareFeature = C["TextCompareFeature"];
export type CofireResponse = C["CofireResponse"];
export type CofirePartner = C["CofirePartner"];

// W8: the features that fired most on the words typed in the search box
export type SearchRequest = C["SearchRequest"];
export type SearchResponse = C["SearchResponse"];
export type SearchFeature = C["SearchFeature"];

// W5: two or three features steered together, each one optionally alone beside it
export type MultiSteerRequest = C["MultiSteerRequest"];
export type MultiSteerResponse = C["MultiSteerResponse"];
export type MultiActivation = C["MultiActivation"];
export type SingleFeatureRun = C["SingleFeatureRun"];
