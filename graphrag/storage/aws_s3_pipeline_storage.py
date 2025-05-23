# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""AWS S3 pipeline storage implementation for GraphRAG."""

import asyncio
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from graphrag.storage.pipeline_storage import PipelineStorage

log = logging.getLogger(__name__)


class S3PipelineStorage(PipelineStorage):
    """AWS S3-based pipeline storage implementation."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        """Initialize S3 pipeline storage.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all objects
            region: AWS region
            access_key_id: AWS access key ID (optional, uses default credentials if None)
            secret_access_key: AWS secret access key (optional)
            session_token: AWS session token (optional)
            endpoint_url: Custom S3 endpoint URL (for S3-compatible services)
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.endpoint_url = endpoint_url

        self._s3_client = None
        self._session = None

    async def _get_s3_client(self):
        """Get or create S3 client."""
        if self._s3_client is None:
            try:
                import boto3
                from botocore.config import Config

                # Create session with credentials if provided
                if self.access_key_id and self.secret_access_key:
                    self._session = boto3.Session(
                        aws_access_key_id=self.access_key_id,
                        aws_secret_access_key=self.secret_access_key,
                        aws_session_token=self.session_token,
                        region_name=self.region,
                    )
                else:
                    # Use default credentials (IAM role, environment variables, etc.)
                    self._session = boto3.Session(region_name=self.region)

                # Configure S3 client
                config = Config(
                    region_name=self.region,
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    max_pool_connections=50,
                )

                self._s3_client = self._session.client(
                    "s3",
                    config=config,
                    endpoint_url=self.endpoint_url,
                )

                log.info(
                    f"Connected to S3 bucket: {self.bucket} in region: {self.region}"
                )

            except ImportError:
                raise ImportError(
                    "boto3 not installed. Install with: pip install boto3"
                )
            except Exception as e:
                log.error(f"Failed to create S3 client: {e}")
                raise

        return self._s3_client

    def _get_s3_key(self, path: str) -> str:
        """Convert local path to S3 key."""
        # Remove leading slash and combine with prefix
        clean_path = path.lstrip("/")
        if self.prefix:
            return f"{self.prefix}/{clean_path}"
        return clean_path

    async def get(
        self, path: str, encoding: Optional[str] = None, as_bytes: bool = False
    ) -> Any:
        """Get file contents from S3."""
        try:
            s3_client = await self._get_s3_client()
            s3_key = self._get_s3_key(path)

            log.debug(f"Getting S3 object: s3://{self.bucket}/{s3_key}")

            # Use asyncio to run boto3 operation in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            )

            # Read content
            content = response["Body"].read()

            if as_bytes:
                return content

            # Decode if encoding specified
            if encoding:
                return content.decode(encoding)

            # Auto-detect encoding for text content
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                # Try common encodings
                for enc in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        return content.decode(enc)
                    except UnicodeDecodeError:
                        continue

                # If all fail, return as bytes
                log.warning(f"Could not decode {path}, returning as bytes")
                return content

        except Exception as e:
            log.error(f"Failed to get S3 object {path}: {e}")
            raise

    async def set(self, path: str, value: Any, encoding: Optional[str] = None) -> None:
        """Put file contents to S3."""
        try:
            s3_client = await self._get_s3_client()
            s3_key = self._get_s3_key(path)

            log.debug(f"Putting S3 object: s3://{self.bucket}/{s3_key}")

            # Convert value to bytes
            if isinstance(value, str):
                content = value.encode(encoding or "utf-8")
            elif isinstance(value, bytes):
                content = value
            else:
                # Assume it's a file-like object or bytes
                if hasattr(value, "read"):
                    content = value.read()
                    if isinstance(content, str):
                        content = content.encode(encoding or "utf-8")
                else:
                    content = str(value).encode(encoding or "utf-8")

            # Upload to S3
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: s3_client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=content,
                    ServerSideEncryption="AES256",  # Enable encryption at rest
                ),
            )

            log.debug(f"Successfully uploaded to S3: s3://{self.bucket}/{s3_key}")

        except Exception as e:
            log.error(f"Failed to put S3 object {path}: {e}")
            raise

    async def has(self, path: str) -> bool:
        """Check if file exists in S3."""
        try:
            s3_client = await self._get_s3_client()
            s3_key = self._get_s3_key(path)

            # Use head_object to check existence
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            )
            return True

        except s3_client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise
        except Exception as e:
            log.error(f"Failed to check S3 object existence {path}: {e}")
            raise

    async def delete(self, path: str) -> None:
        """Delete file from S3."""
        try:
            s3_client = await self._get_s3_client()
            s3_key = self._get_s3_key(path)

            log.debug(f"Deleting S3 object: s3://{self.bucket}/{s3_key}")

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: s3_client.delete_object(Bucket=self.bucket, Key=s3_key)
            )

        except Exception as e:
            log.error(f"Failed to delete S3 object {path}: {e}")
            raise

    async def list(self, prefix: str = "", suffix: str = "") -> list[str]:
        """List files in S3 with optional prefix and suffix filters."""
        try:
            s3_client = await self._get_s3_client()

            # Combine storage prefix with search prefix
            search_prefix = self._get_s3_key(prefix)

            log.debug(f"Listing S3 objects with prefix: {search_prefix}")

            # List objects
            objects = []
            paginator = s3_client.get_paginator("list_objects_v2")

            async def run_paginator():
                for page in paginator.paginate(
                    Bucket=self.bucket, Prefix=search_prefix
                ):
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            key = obj["Key"]
                            # Remove storage prefix to get relative path
                            if self.prefix and key.startswith(self.prefix + "/"):
                                relative_path = key[len(self.prefix) + 1 :]
                            else:
                                relative_path = key

                            # Apply suffix filter
                            if not suffix or relative_path.endswith(suffix):
                                objects.append(relative_path)
                return objects

            return await asyncio.get_event_loop().run_in_executor(None, run_paginator)

        except Exception as e:
            log.error(f"Failed to list S3 objects: {e}")
            raise

    async def get_creation_date(self, path: str) -> Optional[datetime]:
        """Get file creation date from S3 object metadata."""
        try:
            s3_client = await self._get_s3_client()
            s3_key = self._get_s3_key(path)

            # Get object metadata
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            )

            # S3 doesn't have creation date, use last modified
            return response.get("LastModified")

        except Exception as e:
            log.warning(f"Failed to get creation date for {path}: {e}")
            return None

    async def get_file_size(self, path: str) -> Optional[int]:
        """Get file size from S3 object metadata."""
        try:
            s3_client = await self._get_s3_client()
            s3_key = self._get_s3_key(path)

            # Get object metadata
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            )

            return response.get("ContentLength")

        except Exception as e:
            log.warning(f"Failed to get file size for {path}: {e}")
            return None

    async def copy_file(self, source_path: str, dest_path: str) -> None:
        """Copy file within S3."""
        try:
            s3_client = await self._get_s3_client()
            source_key = self._get_s3_key(source_path)
            dest_key = self._get_s3_key(dest_path)

            copy_source = {"Bucket": self.bucket, "Key": source_key}

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=self.bucket,
                    Key=dest_key,
                    ServerSideEncryption="AES256",
                ),
            )

            log.debug(f"Copied S3 object from {source_key} to {dest_key}")

        except Exception as e:
            log.error(
                f"Failed to copy S3 object from {source_path} to {dest_path}: {e}"
            )
            raise

    async def create_presigned_url(self, path: str, expiration: int = 3600) -> str:
        """Create a presigned URL for temporary access to S3 object."""
        try:
            s3_client = await self._get_s3_client()
            s3_key = self._get_s3_key(path)

            url = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": s3_key},
                    ExpiresIn=expiration,
                ),
            )

            return url

        except Exception as e:
            log.error(f"Failed to create presigned URL for {path}: {e}")
            raise

    def get_storage_info(self) -> dict[str, Any]:
        """Get storage configuration information."""
        return {
            "type": "s3",
            "bucket": self.bucket,
            "prefix": self.prefix,
            "region": self.region,
            "endpoint_url": self.endpoint_url,
        }
