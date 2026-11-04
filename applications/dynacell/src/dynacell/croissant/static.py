"""Static fields for Croissant generation.

Holds the prose + identifiers that cannot be derived from manifests:
license, citation, contact, RAI narrative blocks. Every field is
required at instantiation time so a half-filled run fails fast — there
is no silent placeholder fallback.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StaticFields:
    """Immutable bag of authoring-time constants for a Croissant doc.

    Parameters
    ----------
    name
        Top-level ``name`` field for the Croissant doc (e.g.
        ``"DynaCell — A549 (Mantis)"``). Must distinguish this dataset
        from any sibling release docs in the same publication.
    license_url
        Canonical license URL (e.g. ``https://creativecommons.org/licenses/by/4.0/``).
    cite_as
        BibTeX entry for the dataset's primary citation.
    keywords
        Dataset-specific ``keywords`` entries, appended to the shared modality
        terms every DynaCell release carries. Cell line and platform belong
        here — they differ per release and must not be hardcoded in the builder.
    creators
        Top-level ``creator`` block, verbatim JSON-LD entries. This is the field
        aggregators read for attribution, so the list must name the institutions
        that produced *this* subset's data, not the release as a whole; upstream
        provenance additionally goes in ``prov_was_derived_from``.
    publisher_name, publisher_url
        Top-level ``publisher`` block.
    contact_email
        Contact for dataset issues.
    aws_bucket
        S3 bucket name (without ``s3://``) hosting the OZX artifacts.
    aws_prefix
        Key prefix inside the bucket (e.g. ``"dynacell/v1"``).
    rai_data_collection
        Prose for ``rai:dataCollection`` — how the data was acquired.
    rai_data_biases
        Prose for ``rai:dataBiases`` — known sampling/scope biases.
    rai_annotations_per_item
        Prose for ``rai:annotationsPerItem`` — what's annotated per FOV.
    rai_personal_sensitive_information
        Prose for ``rai:personalSensitiveInformation``.
    rai_data_limitations
        Prose for ``rai:dataLimitations`` — known limitations / caveats.
        Used in place of the non-existent ``rai:ethicalReview``.
    rai_data_use_cases
        Prose for ``rai:dataUseCases`` — validated tasks and out-of-scope use.
    rai_data_social_impact
        Prose for ``rai:dataSocialImpact`` — positive impact, risks, mitigations.
    prov_was_derived_from
        Structured ``prov:wasDerivedFrom`` entries — list of upstream
        sources this release is derived from (PROV-O). Each entry is a
        dict with at minimum ``@id``, ``prov:label``, ``description``,
        and optionally ``sc:license`` and ``prov:wasAttributedTo``.
        See NeurIPS RAI guidelines for the exact shape.
    prov_was_generated_by
        Structured ``prov:wasGeneratedBy`` entries — list of
        ``prov:Activity`` dicts describing collection / preprocessing /
        annotation steps, each with ``@type``, ``prov:label``,
        ``description``, and ``prov:wasAttributedTo`` listing the
        agents / software involved.
    """

    name: str
    license_url: str
    cite_as: str
    keywords: tuple[str, ...]
    creators: tuple[dict[str, object], ...]
    publisher_name: str
    publisher_url: str
    contact_email: str
    aws_bucket: str
    aws_prefix: str
    rai_data_collection: str
    rai_data_biases: str
    rai_annotations_per_item: str
    rai_personal_sensitive_information: str
    rai_data_limitations: str
    rai_data_use_cases: str
    rai_data_social_impact: str
    rai_has_synthetic_data: bool
    prov_was_derived_from: list[dict[str, object]]
    prov_was_generated_by: list[dict[str, object]]

    def __post_init__(self) -> None:
        """Reject any None / empty-string / empty-list field at construction time.

        Boolean fields (``rai_has_synthetic_data``) are exempt: ``False``
        is a meaningful value distinct from "unset".
        """
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, bool):
                continue
            if value is None or value == "" or value == []:
                raise ValueError(
                    f"StaticFields.{field_name} is required and must be non-empty; "
                    "fill in real values before generating Croissant"
                )
